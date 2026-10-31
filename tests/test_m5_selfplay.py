"""M5 单测：owner-1 自博弈动作解码、obs1 打包、快照刷新。"""
import sys

sys.path.insert(0, r'D:\pynoob\FirstLight\FirstLight_CR')
sys.path.insert(0, r'D:\pynoob\r2dreamer')

import numpy as np
import torch
from tensordict import TensorDict

from envs.firstlight import ABILITY_INDEX, ACTION_DIM, CELLS, OPP_OWNER, FirstLight, FirstLightInlineEnv
from native_runner.contracts import ActionV1, ActionKind


def _env_with_opp():
    env = object.__new__(FirstLight)
    env._opp_policy = 'selfplay'
    env._action_v1 = ActionV1
    env._opp_plays = 0
    env._opp_last = 'init'
    env._ability_by_entity = {}
    env._ability_by_entity_1 = {777: 'Deflect'}
    return env


def _mask1():
    class Mask:
        hand_slots = [True, True, False, False]
        placement_masks = {
            '0': {'row_major': [[True] * 18 for _ in range(32)]},
            '1': {'row_major': [[True] * 18 for _ in range(32)]},
        }
        ability_sources = (777,)
    return Mask()


def _native_now():
    # 座位翻转后：快照对手 = owner-0，手牌数据挂 owner-0；agent = owner-1
    return {"players": [
        {"owner": 0, "hand": [{"cardId": 26000069, "cost": 4, "handIndex": 0},
                              {"cardId": 26000072, "cost": 4, "handIndex": 1},
                              {"cardId": 26000000, "cost": 3, "handIndex": 2},
                              {"cardId": 26000003, "cost": 5, "handIndex": 3}]},
        {"owner": 1, "hand": []},
    ]}


class _Battle:
    def __init__(self):
        self.mask = _mask1()
    def action_mask(self, owner):
        assert owner == OPP_OWNER  # 座位翻转后：对手 = owner-0
        return self.mask


def test_selfplay_action_decode():
    env = _env_with_opp()
    env._battle = _Battle()
    native_now = _native_now()

    # None → 脚本热身（随机合法动作或 skip）
    a = env._selfplay_action(None, native_now)
    assert a is None or a.kind in (ActionKind.WAIT, ActionKind.PLAY_CARD)

    # WAIT（索引 0）
    a = env._selfplay_action(0, native_now)
    assert a.kind == ActionKind.WAIT and a.owner == OPP_OWNER

    # 出牌：slot0 × cell(3,4) → play owner=1, hand_slot=0
    a = env._selfplay_action(1 + 0 * CELLS + 4 * 18 + 3, native_now)
    assert a.kind == ActionKind.PLAY_CARD and a.owner == OPP_OWNER
    assert a.hand_slot == 0 and a.card_id == 26000069 and a.target_grid == (3, 4)

    # 非法槽（slot2 hand_slots=False）→ 降级 WAIT
    a = env._selfplay_action(1 + 2 * CELLS, native_now)
    assert a.kind == ActionKind.WAIT and env._opp_last == 'selfplay_illegal'

    # 技能位 → ACTIVATE_ABILITY owner=1（实体/技能来自 owner-1 快照表）
    a = env._selfplay_action(ABILITY_INDEX, native_now)
    assert a.kind == ActionKind.ACTIVATE_ABILITY and a.owner == OPP_OWNER
    assert a.source_entity == 777 and a.ability_id == 'Deflect'
    print('PASS: _selfplay_action decode (wait/play/illegal/ability/warmup)')


def test_pop_obs1():
    env = object.__new__(FirstLightInlineEnv)
    env._torch = torch
    env._TensorDict = TensorDict
    env.env_num = 2
    env._last_obs1 = [
        {"image": np.zeros((4, 4, 3), dtype=np.uint8), "elixir": np.array([1.0], dtype=np.float32),
         "hand_ids": np.zeros(4, dtype=np.int64), "deck_ids": np.zeros(8, dtype=np.int64),
         "action_mask": np.ones(ACTION_DIM, dtype=np.uint8), "is_first": np.bool_(True)},
        {"image": np.zeros((4, 4, 3), dtype=np.uint8), "elixir": np.array([2.0], dtype=np.float32),
         "hand_ids": np.zeros(4, dtype=np.int64), "deck_ids": np.zeros(8, dtype=np.int64),
         "action_mask": np.ones(ACTION_DIM, dtype=np.uint8), "is_first": np.bool_(True)},
    ]
    td = env.pop_obs1()
    assert td is not None and td.batch_size == (2,)
    assert td["elixir"].shape == (2, 1) and td["image"].shape == (2, 4, 4, 3)
    assert env.pop_obs1() is None  # 取走即清
    env._last_obs1 = [None, env._last_obs1[0]]
    assert env.pop_obs1() is None  # 任一缺失即 None（自愈降级脚本）
    print('PASS: pop_obs1 stacking/clear/None-degrade')


def test_snapshot_refresh():
    import copy
    import gymnasium as gym
    from omegaconf import OmegaConf
    import hydra
    from dreamer import Dreamer

    with hydra.initialize(version_base=None, config_path='../configs'):
        cfg = hydra.compose(config_name='configs',
                            overrides=['env=dmc_vision', 'model=size12M', 'device=cpu', 'model.compile=false'])
    from gymnasium.spaces import Discrete
    obs_space = gym.spaces.Dict({
        "image": gym.spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8),
        "elixir": gym.spaces.Box(0, 10, (1,), dtype=__import__('numpy').float32),
        "elixir_rate": gym.spaces.Box(0, 1, (1,), dtype=__import__('numpy').float32),
        "hand_ids": gym.spaces.Box(0, 300, (4,), dtype=__import__('numpy').int64),
        "deck_ids": gym.spaces.Box(0, 300, (8,), dtype=__import__('numpy').int64),
        "action_mask": gym.spaces.Box(0, 1, (2306,), dtype=np.uint8),
    })
    agent = Dreamer(cfg.model, obs_space, gym.spaces.Discrete(2306)).to('cpu')
    opp = copy.deepcopy(agent)
    opp.eval()
    # 模拟 refresh：load_state_dict 后参数逐位一致
    opp.load_state_dict(agent.state_dict())
    for (k1, p1), (k2, p2) in zip(agent.state_dict().items(), opp.state_dict().items()):
        assert k1 == k2 and torch.equal(p1, p2), k1
    # 对手推理路径：owner-1 视图观测 → 扁平动作
    B = 2
    # 与 InlineEnv.step 打包约定一致：标量键 (B,1)，多维键不 lift
    obs1 = TensorDict({
        "image": torch.randint(0, 255, (B, 64, 64, 3), dtype=torch.uint8),
        "elixir": torch.rand(B, 1), "elixir_rate": torch.rand(B, 1),
        "hand_ids": torch.randint(1, 60, (B, 4)), "deck_ids": torch.randint(1, 60, (B, 8)),
        "action_mask": torch.ones(B, 2306), "is_first": torch.ones(B, 1, dtype=torch.bool),
    }, batch_size=(B,))
    state = opp.get_initial_state(B)
    with torch.no_grad():
        a1, state2 = opp.act(obs1, state, eval=True)
    assert a1.shape == (B, 2306) and state2["prev_action"].shape == (B, 2306)
    print('PASS: snapshot deepcopy/load_state_dict/act(obs1)')


def test_random_paired():
    """random_paired：奇偶局同 seed 成对换边（A/B 臂升级主线的采样器）。"""
    env = object.__new__(FirstLight)
    env._deck_mode = 'random_paired'
    env._episodes = 1
    env._pair_prev = None
    env._deck_override = None
    env._seed_override = None
    env._deck0 = env._deck1 = None

    # 奇数局：正常抽并寄存
    assert env._sample_decks(123) is True
    assert env._deck0 is not None and len(env._deck0) == 8
    assert env._pair_prev == (123, env._deck0, env._deck1)
    dA, dB = env._deck0, env._deck1

    # 偶数局：不重抽（返回 False），reset 套用交换对
    env._episodes = 2
    assert env._sample_decks(999) is False   # 忽略新 seed
    seed, (d0, d1) = env._pair_prev[0], (env._pair_prev[2], env._pair_prev[1])
    assert seed == 123 and d0 == dB and d1 == dA
    env._pair_prev = None                    # 配对完成

    # 下一对的奇数局重新独立抽
    env._episodes = 3
    assert env._sample_decks(456) is True
    assert env._deck0 != dA or env._deck1 != dB  # 新 pair 独立
    print('PASS: random_paired odd/even seed-deck pairing')


if __name__ == '__main__':
    test_selfplay_action_decode()
    test_pop_obs1()
    test_snapshot_refresh()
    test_random_paired()
    print('ALL M5 TESTS PASSED')
