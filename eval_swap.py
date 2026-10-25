"""换边双打 A 计算 harness。

对每对卡组 (D1, D2) 打一对同 seed 换边局：
  局 A: deck0=D1 (红), deck1=D2
  局 B: deck0=D2 (红), deck1=D1
  A(D1, D2) = (r_A - r_B) / 2
  r_A/r_B 分别是局 A/局 B 里 player-0 座位的终局奖励（±1 胜负 / 0 平）。
同 seed + 座位互换把先手/座次偏置对消，得到零和的卡组优势估计。

用法：
  python eval_swap.py --run-dir runs/firstlight_m3 --pairs 2
要求 run-dir 里有 latest.pt（trainer 周期保存）和 .hydra/config.yaml。
运行期间会独占两套离线会话的引擎——先停掉训练。
"""

import argparse
import pathlib
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import tools  # noqa: E402
from dreamer import Dreamer  # noqa: E402
from envs import make_env  # noqa: E402
from envs.firstlight import DECK_POOL, FirstLightInlineEnv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--pairs", type=int, default=2, help="评估的卡组对数")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-steps", type=int, default=900, help="单局步数保险上限")
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run_dir)
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    tools.set_seed_everywhere(cfg.seed)
    device = args.device

    def make_fn(idx):
        return make_env(cfg.env, idx)

    envs = FirstLightInlineEnv(make_fn, device=device, env_num=2)

    def inner_firstlight(w):
        while hasattr(w, "env"):
            w = w.env
        return w

    fl = [inner_firstlight(e) for e in envs._envs]
    agent = Dreamer(cfg.model, envs.observation_space, envs.action_space).to(device)
    ckpt = torch.load(run_dir / "latest.pt", map_location=device, weights_only=False)
    agent.load_state_dict(ckpt["agent_state_dict"])
    agent.eval()
    print(f"checkpoint loaded: {run_dir / 'latest.pt'}")

    rng = np.random.default_rng(cfg.seed + 17)
    table = []
    for pair in range(args.pairs):
        d1 = tuple(int(c) for c in rng.choice(DECK_POOL, size=8, replace=False))
        d2 = tuple(int(c) for c in rng.choice(DECK_POOL, size=8, replace=False))
        seed = int(rng.integers(1, 10 ** 6))
        # env0: D1 执红；env1: D2 执红（同 seed、座位互换）
        fl[0].set_next_match(d1, d2, seed)
        fl[1].set_next_match(d2, d1, seed)
        print(f"\n== pair {pair + 1}/{args.pairs}: seed={seed} ==")
        print(f"  env0 deck0={d1}\n  env1 deck0={d2}")

        agent_state = agent.get_initial_state(2)
        act = agent_state["prev_action"].clone()
        done = torch.ones(2, dtype=torch.bool, device=device)
        episode_ret = [0.0, 0.0]
        pair_r = [None, None]
        lengths = [0, 0]
        steps = 0
        while steps < args.max_steps and any(v is None for v in pair_r):
            td, step_done = envs.step(act.detach(), done)
            r = td["reward"].reshape(-1).tolist()
            for i in (0, 1):
                if pair_r[i] is None:
                    episode_ret[i] += r[i]
                    lengths[i] += 1
                if bool(step_done[i]) and pair_r[i] is None:
                    pair_r[i] = episode_ret[i]
                    print(f"  env{i} episode done: ret={pair_r[i]:+.0f} length={lengths[i]}")
            done = step_done.to(device)
            if all(v is not None for v in pair_r):
                break
            act, agent_state = agent.act(td.to(device).clone(), agent_state, eval=True)
            steps += 1

        if any(v is None for v in pair_r):
            print("  pair incomplete (max steps); skipped")
            continue
        a = (pair_r[0] - pair_r[1]) / 2
        table.append({"d1": d1, "d2": d2, "seed": seed, "rA": pair_r[0], "rB": pair_r[1], "A": a})
        print(f"  A(D1,D2) = ({pair_r[0]:+.0f} - ({pair_r[1]:+.0f})) / 2 = {a:+.1f}")

    print("\n===== 换边双打优势表 =====")
    for row in table:
        names = lambda d: ",".join(str(c)[-2:] for c in d)
        print(f"A(D1={names(row['d1'])} vs D2={names(row['d2'])}) = {row['A']:+.1f}  "
              f"(rA={row['rA']:+.0f}, rB={row['rB']:+.0f}, seed={row['seed']})")
    print("语义：A=+1 D1 两座全胜；A=-1 D1 两座全败；0 = 平分或双平。")


if __name__ == "__main__":
    main()
