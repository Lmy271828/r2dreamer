"""League ③④⑤：候选池 / 爬山选卡 / 门禁（docs/deck_selection.md）。

- GateChecker: 双门判定（league_stats 成熟度 + KL 探针序列 + WM 平台化）。
  数据不足/门未过 → shadow 模式（只算不干预）。
- hill_climb: 单卡置换爬山（docs §4），打分回调注入（readout/rollout 可换）。
- DeckSelector: 每 selector_every 局运行一次——建池 → 爬山 → 产出
  (deck0, deck1) 或 shadow 排序；ε 探索与 PFSP 权重按 deck_selection §5。

启用纪律（deck_selection §5 双门）：GateChecker 不通过时 selector 一律
shadow——本模块不做门禁绕行。
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------- 门禁 ----

class GateChecker:
    """双门判定。probes: [(updates, kl), ...]（KL 平台期探针序列，可为空）。

    门一（策略 KL）：最近 3 个间隔的 KL 全部 < kl_eps。
    门二（WM 平台化）：外部传入 loss/dyn、loss/rew 近段斜率（由调用方从
    metrics 取，本类不读盘）。
    """

    def __init__(self, kl_eps: float = 0.01, kl_windows: int = 3, min_games: int = 10):
        self.kl_eps = kl_eps
        self.kl_windows = kl_windows
        self.min_games = min_games

    def policy_gate(self, probes: list[tuple[int, float]]) -> tuple[bool, str]:
        kls = [kl for _, kl in probes[-self.kl_windows:]]
        if len(kls) < self.kl_windows:
            return False, f"insufficient probes ({len(kls)}/{self.kl_windows})"
        if all(k < self.kl_eps for k in kls):
            return True, "policy KL plateau"
        return False, f"policy KL not plateaued: {['%.4f' % k for k in kls]}"

    def data_gate(self, stats_rows: list[dict]) -> tuple[bool, str]:
        agg = defaultdict(lambda: {"n": 0})
        for r in stats_rows:
            if r.get("reason") == "aborted":
                continue
            agg[tuple(sorted(r["opp_deck"]))]["n"] += 1
        mature = [d for d, v in agg.items() if v["n"] >= self.min_games]
        if not mature:
            return False, f"no opponent deck with >={self.min_games} games"
        return True, f"{len(mature)} mature opponent decks"

    def check(self, probes, stats, wm_plateau: bool) -> tuple[bool, list[str]]:
        reasons = []
        ok1, r1 = self.policy_gate(probes)
        if not ok1:
            reasons.append(r1)
        ok2, r2 = self.data_gate(stats)
        if not ok2:
            reasons.append(r2)
        if not wm_plateau:
            reasons.append("world model not plateaued (loss/dyn or rew rising)")
        return (not reasons), reasons


# ------------------------------------------------------------------ 池 ----

def build_pool(stats_rows: list[dict], min_games: int = 10) -> dict:
    """league_stats 行 → {"strong": [...], "diverse": [...]}（aborted 过滤）。"""
    agg = defaultdict(lambda: {"n": 0, "wins": 0})
    for r in stats_rows:
        if r.get("reason") == "aborted":
            continue
        key = tuple(sorted(r["opp_deck"]))
        agg[key]["n"] += 1
        agg[key]["wins"] += 1 if r.get("win") else 0
    strong = sorted(
        ({"deck": list(k), "games": v["n"], "win_rate": v["wins"] / v["n"]}
         for k, v in agg.items() if v["n"] >= min_games),
        key=lambda x: -x["win_rate"])
    diverse = [{"deck": list(k), "games": v["n"]} for k, v in agg.items()]
    return {"strong": strong, "diverse": diverse}


# ----------------------------------------------------------------- 爬山 ----

def hill_climb(start_deck: list[int], score_fn, candidates: list[int],
               max_steps: int = 8, rng: random.Random | None = None):
    """单卡置换爬山（docs/deck_selection.md §4）。

    score_fn(deck) -> float（标量评分）；邻居 = 任一槽位 × 任一未用候选卡
    （v1 无槽位限制——800+ 局随机排布从未被引擎拒绝，经验记录）。
    返回 (best_deck, best_value, path)。
    """
    rng = rng or random.Random()
    best = list(start_deck)
    best_v = score_fn(best)
    path = [{"step": 0, "deck": list(best), "value": best_v}]
    for step in range(1, max_steps + 1):
        neighbors = []
        seen = {tuple(best)}
        for slot in range(8):
            for c in candidates:
                if c in best:
                    continue
                cand = list(best)
                cand[slot] = c
                key = tuple(sorted(cand))
                if key in seen:
                    continue
                seen.add(key)
                neighbors.append(cand)
        rng.shuffle(neighbors)
        improved = None
        for cand in neighbors:
            v = score_fn(cand)
            if v > best_v + 1e-9:
                if improved is None or v > improved[1]:
                    improved = (cand, v)
        if improved is None:
            break
        best, best_v = improved[0], improved[1]
        path.append({"step": step, "deck": list(best), "value": best_v})
    return best, best_v, path


# ------------------------------------------------------------- 选卡器 ----

class DeckSelector:
    """每 selector_every 局运行：门禁 → 池 → 爬山 → (deck0, deck1) 或 shadow。

    score_fn 由调用方注入（score_decks 的 readout/rollout LCB），
    本类不持有模型。shadow 模式只产出排序日志，不改变对局卡组。
    """

    def __init__(self, epsilon: float = 0.2, min_games: int = 10,
                 max_steps: int = 8, seed: int = 0):
        self.gates = GateChecker(min_games=min_games)
        self.epsilon = epsilon
        self.max_steps = max_steps
        self.rng = random.Random(seed)
        self.rounds = 0

    def select(self, stats_rows: list[dict], probes: list[tuple[int, float]],
               wm_plateau: bool, score_fn, agent_deck: list[int]):
        """返回 dict(mode, deck0, deck1, info)。deck1 = PFSP 加权抽的对手卡组。"""
        self.rounds += 1
        pool = build_pool(stats_rows)
        ok_data, r_data = self.gates.data_gate(stats_rows)
        ok_pol, r_pol = self.gates.policy_gate(probes)
        gates_ok = ok_data and ok_pol and wm_plateau
        reasons = ([r for ok, r in ((ok_pol, r_pol), (ok_data, r_data)) if not ok]
                   + ([] if wm_plateau else ["wm not plateaued"]))
        info = {"round": self.rounds, "gates_ok": gates_ok, "reasons": reasons}

        if not gates_ok:
            info["mode"] = "shadow"
            return {"mode": "shadow", "info": info}

        # 爬山：从胜率最高的强池卡组出发
        if pool["strong"]:
            start = pool["strong"][0]["deck"]
        else:
            start = agent_deck
        best, best_v, path = hill_climb(start, score_fn,
                                        candidates=sorted({c for d in pool["diverse"] for c in d["deck"]}),
                                        max_steps=self.max_steps, rng=self.rng)
        info["path"] = path

        # PFSP 加权抽对手卡组（真实战绩：劣势 + 少样本不确定性加成）
        opp_weights = []
        for d in pool["diverse"]:
            key = tuple(sorted(d["deck"]))
            row = next((s for s in stats_rows if tuple(sorted(s["opp_deck"])) == key), None)
            win = 1.0 if (row and row.get("win")) else 0.0
            opp_weights.append((d, 0.1 + (1.0 - win)))
        total = sum(w for _, w in opp_weights)
        deck1 = self.rng.choices([d["deck"] for d, _ in opp_weights],
                                 weights=[w / total for _, w in opp_weights])[0]

        if self.rng.random() < self.epsilon:  # ε 探索档：我方卡组从多样池抽
            deck0 = list(self.rng.choice(pool["diverse"])["deck"])
            info["mode"] = "active_explore"
        else:
            deck0 = best
            info["mode"] = "active"
        info["hill_best_value"] = best_v
        return {"mode": info["mode"], "deck0": deck0, "deck1": deck1, "info": info}
