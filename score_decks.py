"""V(s_open, deck) 批量打分 stub：给定 checkpoint + 候选卡组 → 排序。

用法:
  python score_decks.py --checkpoint runs/1030_172220_m5_selfplay_paired_b/latest.pt \
      --random 8                       # 随机抽 8 套候选
  python score_decks.py --checkpoint ... --decks-json decks.json
      # decks.json: [[cardid×8], ...]（原生 card id）

开局态: 默认取 --frame 指定的战场截图（ARENA_CROP 内、任意时刻均可——
所有候选共用同一状态，排序只依赖卡组条件；绝对值的校准靠真实战绩门禁）。
侧通道: 圣水 5.0（开局标准值）、开局增速；对手条件 = UNK（边缘模式
V(s,d0)，即元分析读数；--conditional 切换完整 matchup 模式）。

输出: 终端排序表 + <checkpoint目录>/deck_scores.json。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "FirstLight" / "FirstLight_CR"))

ARENA_CROP = (111, 430, 967, 1620)
SIZE = (176, 128)  # (H, W) —— 与 firstlight_swarm.yaml 的 size 一致


def load_agent(checkpoint: Path, device: str = "cpu"):
    import hydra
    from dreamer import Dreamer
    import gymnasium as gym

    with hydra.initialize(version_base=None, config_path="configs", job_name="score_decks"):
        cfg = hydra.compose(config_name="configs", overrides=[
            "env=dmc_vision", "model=size12M", f"device={device}",
            "model.compile=false", "model.rep_loss=r2dreamer",
        ])
    obs_space = gym.spaces.Dict({
        "image": gym.spaces.Box(0, 255, (176, 128, 3), dtype=np.uint8),
        "elixir": gym.spaces.Box(0, 10, (1,), dtype=np.float32),
        "elixir_rate": gym.spaces.Box(0, 1, (1,), dtype=np.float32),
        "hand_ids": gym.spaces.Box(0, 300, (4,), dtype=np.int64),
        "deck_ids": gym.spaces.Box(0, 300, (8,), dtype=np.int64),
        "opp_deck_ids": gym.spaces.Box(0, 300, (8,), dtype=np.int64),
        "action_mask": gym.spaces.Box(0, 1, (2306,), dtype=np.uint8),
    })
    agent = Dreamer(cfg.model, obs_space, gym.spaces.Discrete(2306)).to(device)
    blob = torch.load(checkpoint, map_location=device, weights_only=False)
    agent.load_state_dict(blob["agent_state_dict"])
    agent.eval()
    for p in agent.parameters():
        p.requires_grad_(False)
    return agent


def opening_feat(agent, frame_png: bytes | None):
    """开局态特征：frame → encoder → RSSM 后验一步（prev_action=WAIT）。"""
    import io
    from PIL import Image
    from native_runner.vision.capture import scale_crop, crop_resize_rect

    if frame_png is None:
        raise SystemExit("需要 --frame <战场截图png>（ARENA_CROP 已含则整图亦可）")
    img = Image.open(io.BytesIO(frame_png)).convert("RGB")
    rect = scale_crop(ARENA_CROP, img.size)
    arena = crop_resize_rect(frame_png, rect, SIZE)             # (H,W,3) uint8
    obs = {
        "image": torch.from_numpy(arena).unsqueeze(0),          # (1,H,W,C)
        "elixir": torch.tensor([[5.0]]),                        # 开局标准值
        "elixir_rate": torch.tensor([[2.5 / 60.0]]),            # tick0 增速（1x 换算）
    }
    with torch.no_grad():
        embed = agent.encoder(obs)                              # (1,E)
        stoch, deter = agent.rssm.initial(1)
        reset = torch.zeros(1, dtype=torch.bool)
        prev_action = torch.zeros(1, agent.act_dim)
        stoch, deter, _ = agent.rssm.obs_step(stoch, deter, prev_action, embed, reset)
        return agent.rssm.get_feat(stoch, deter)                # (1,F)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--frame", type=Path, help="战场截图 png（缺省则拒绝运行）")
    ap.add_argument("--decks-json", type=Path, help="候选卡组 [[cardid×8], ...]")
    ap.add_argument("--random", type=int, default=0, help="另加 N 套随机卡组")
    ap.add_argument("--conditional", action="store_true",
                    help="完整 matchup 模式（需 --opp-json）；缺省 = 边缘模式 V(s,d0)")
    ap.add_argument("--opp-json", type=Path)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from envs.firstlight import DECK_INDEX, DECK_POOL
    from native_runner.card_names import CARD_NAMES
    from native_runner.vision.capture import capture_png

    frame = capture_png("adb", "127.0.0.1:16416") if args.frame is None else args.frame.read_bytes()
    if args.frame is None:
        print("提示: 未指定 --frame，已现场抓取 guest 帧（若 guest 不在对局/战场画面，排序仅供管线联调）")

    decks: list[tuple[int, ...]] = []
    if args.decks_json:
        decks += [tuple(int(c) for c in d) for d in json.loads(args.decks_json.read_text())]
    rng = np.random.default_rng(0)
    for _ in range(args.random):
        decks.append(tuple(int(c) for c in rng.choice(DECK_POOL, size=8, replace=False)))
    if not decks:
        raise SystemExit("没有候选卡组（--decks-json / --random）")
    assert all(len(d) == 8 and len(set(d)) == 8 and set(d) <= set(DECK_POOL) for d in decks), \
        "卡组必须为 8 张互不重复的白名单卡"

    agent = load_agent(args.checkpoint, args.device)
    feat = opening_feat(agent, frame).repeat(len(decks), 1)         # (N,F)
    d_ids = torch.tensor([[DECK_INDEX[c] for c in d] for d in decks], dtype=torch.long)
    o_ids = torch.zeros_like(d_ids)                                 # UNK = 边缘模式
    if args.conditional:
        if not args.opp_json:
            raise SystemExit("--conditional 需要 --opp-json（与候选一一对应的对手卡组）")
        opps = json.loads(args.opp_json.read_text())
        o_ids = torch.tensor([[DECK_INDEX[c] for c in d] for d in opps], dtype=torch.long)

    with torch.no_grad():
        values = agent.value(feat, d_ids, o_ids, drop_opp=None).mode().squeeze(-1)

    order = torch.argsort(values, descending=True)
    name = lambda cid: CARD_NAMES.get(cid, str(cid))
    print(f"{'rank':>4} {'V(s_open,d)':>12}  deck")
    out = []
    for rk, i in enumerate(order.tolist(), 1):
        cards = ", ".join(name(c) for c in decks[i])
        print(f"{rk:>4} {values[i].item():>12.5f}  [{cards}]")
        out.append({"rank": rk, "value": values[i].item(), "deck": list(decks[i])})
    dest = args.checkpoint.parent / "deck_scores.json"
    dest.write_text(json.dumps({"checkpoint": str(args.checkpoint), "scores": out},
                               ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"saved {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
