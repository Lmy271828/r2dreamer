"""从 league_stats.jsonl 构建候选卡组池（deck_selection.md §3）。

用法:
  python build_deck_pool.py --stats <run>/t0/env_26789/league_stats.jsonl \
      [--min-games 10] [--out pool.json]

结构: {"strong": [...], "diverse": [...], "meta": {...}}
数据不足（strong 为空）时调用方应退化为均匀随机池（当前行为）。
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stats", type=Path, required=True)
    ap.add_argument("--min-games", type=int, default=10)
    ap.add_argument("--out", type=Path, default=Path("deck_pool.json"))
    args = ap.parse_args()

    recs = [json.loads(l) for l in args.stats.read_text(encoding="utf-8").splitlines() if l.strip()]
    recs = [r for r in recs if r.get("reason") != "aborted"]  # 事故局过滤
    agg = defaultdict(lambda: {"n": 0, "wins": 0})
    for r in recs:
        key = json.dumps(sorted(r["opp_deck"]))  # 对手卡组以排序元组为身份
        agg[key]["n"] += 1
        agg[key]["wins"] += 1 if r["win"] else 0

    strong = [
        {"deck": json.loads(k), "games": v["n"], "win_rate": v["wins"] / v["n"]}
        for k, v in agg.items() if v["n"] >= args.min_games
    ]
    strong.sort(key=lambda x: -x["win_rate"])
    diverse = [{"deck": json.loads(k), "games": v["n"]} for k, v in agg.items()]

    pool = {
        "strong": strong,
        "diverse": diverse,
        "meta": {"episodes": len(recs), "unique_opponent_decks": len(agg),
                 "min_games": args.min_games},
    }
    args.out.write_text(json.dumps(pool, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"pool: strong={len(strong)} diverse={len(diverse)} "
          f"(from {len(recs)} episodes, aborted excluded)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
