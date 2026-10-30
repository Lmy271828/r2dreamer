"""S4 单测：DeckEncoder 槽位敏感性、条件化 value 双模式、Dreamer 级前向。"""
import sys

sys.path.insert(0, r'D:\pynoob\FirstLight\FirstLight_CR')
sys.path.insert(0, r'D:\pynoob\r2dreamer')

import numpy as np
import torch

from networks import DeckCondValueHead, DeckEncoder, default_card_table

torch.manual_seed(0)


def test_slot_sensitivity():
    enc = DeckEncoder(default_card_table())
    d = torch.randint(1, 60, (1, 8))
    v1 = enc(d)
    d2 = d.clone()
    d2[0, 0], d2[0, 1] = d[0, 1].item(), d[0, 0].item()  # 交换前两槽（英雄槽区）
    v2 = enc(d2)
    assert not torch.allclose(v1, v2, atol=1e-5), "slot swap ignored -- slot encoding dead"
    # 同卡组不同槽位排列 → 输出随槽位变化（位置编码生效）
    perms = torch.stack([d[0, p] for p in torch.randperm(8)]).unsqueeze(0)
    assert not torch.allclose(enc(d), enc(perms), atol=1e-5)
    print('PASS: DeckEncoder slot sensitivity')


def test_two_modes():
    class Cfg:
        name = "critic"; layers = 2; units = 64; act = "ELU"
        symlog_inputs = False; outscale = 1.0; device = "cpu"
        shape = (32,); dist = None
    import types
    Cfg.dist = types.SimpleNamespace(unimix_ratio=0.0, name="symexp_twohot", bin_num=32)
    head = DeckCondValueHead(Cfg, feat_dim=128, card_table=default_card_table())
    B = 4
    feat = torch.randn(B, 128)
    d0 = torch.randint(1, 60, (B, 8)); d1 = torch.randint(1, 60, (B, 8))
    drop = torch.tensor([True, False, True, False])
    dist = head(feat, d0, d1, drop_opp=drop)
    # 条件 vs 丢弃模式：drop=True 的序列在换对手卡组时 value 不变（UNK 屏蔽）
    d1_alt = d1.clone(); d1_alt[0] = torch.randint(1, 60, (8,))
    feat0 = feat[0:1]
    v_orig = head(feat0, d0[0:1], d1[0:1], drop_opp=torch.tensor([True])).mode()
    v_alt = head(feat0, d0[0:1], d1_alt[0:1], drop_opp=torch.tensor([True])).mode()
    assert torch.allclose(v_orig, v_alt, atol=1e-5), "dropout mode leaks opponent deck"
    # 条件模式（drop=False）换对手卡组 → value 变化
    v_orig2 = head(feat0, d0[0:1], d1[0:1], drop_opp=torch.tensor([False])).mode()
    v_alt2 = head(feat0, d0[0:1], d1_alt[0:1], drop_opp=torch.tensor([False])).mode()
    assert not torch.allclose(v_orig2, v_alt2, atol=1e-5), "conditional mode ignores opponent deck"
    # (B,T,F) 形状路径
    dist3 = head(feat.unsqueeze(1), d0, d1, drop_opp=drop)
    assert dist3.mode().shape[:2] == (B, 1)
    print('PASS: DeckCondValueHead dropout/conditional modes + shape paths')


def test_dreamer_smoke():
    import gymnasium as gym
    from omegaconf import OmegaConf
    import hydra
    from dreamer import Dreamer
    from tensordict import TensorDict
    with hydra.initialize(version_base=None, config_path='../configs'):
        cfg = hydra.compose(config_name='configs',
                            overrides=['env=dmc_vision', 'model=size12M', 'device=cpu', 'model.compile=false'])
    obs_space = gym.spaces.Dict({
        "image": gym.spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8),
        "elixir": gym.spaces.Box(0, 10, (1,), dtype=np.float32),
        "elixir_rate": gym.spaces.Box(0, 1, (1,), dtype=np.float32),
        "hand_ids": gym.spaces.Box(0, 300, (4,), dtype=np.int64),
        "deck_ids": gym.spaces.Box(0, 300, (8,), dtype=np.int64),
        "opp_deck_ids": gym.spaces.Box(0, 300, (8,), dtype=np.int64),
        "action_mask": gym.spaces.Box(0, 1, (2306,), dtype=np.uint8),
    })
    agent = Dreamer(cfg.model, obs_space, gym.spaces.Discrete(2306)).to('cpu')
    assert agent._cond_value and isinstance(type(agent.value), object)
    B, T = 2, 8
    mask = torch.zeros(B, T, 2306); mask[..., 0] = 1; mask[..., 1:577] = 1
    data = TensorDict({
        'image': torch.randint(0, 255, (B, T, 64, 64, 3), dtype=torch.uint8),
        'elixir': torch.rand(B, T, 1), 'elixir_rate': torch.rand(B, T, 1),
        'hand_ids': torch.randint(1, 60, (B, T, 4)), 'deck_ids': torch.randint(1, 60, (B, T, 8)),
        'opp_deck_ids': torch.randint(1, 60, (B, T, 8)),
        'action_mask': mask, 'action': torch.zeros(B, T, 2306), 'reward': torch.zeros(B, T, 1),
        'is_first': torch.ones(B, T, 1, dtype=torch.bool),
        'is_last': torch.zeros(B, T, 1, dtype=torch.bool),
        'is_terminal': torch.zeros(B, T, 1, dtype=torch.bool),
    }, batch_size=(B, T))
    p = agent.preprocess(data)
    s0, d0 = agent.rssm.initial(B)
    (st, de), mets = agent._cal_grad(p, (s0, d0))
    assert torch.isfinite(torch.tensor([float(mets['loss/value'])]))
    print('PASS: Dreamer _cal_grad with conditional value')


if __name__ == '__main__':
    test_slot_sensitivity()
    test_two_modes()
    test_dreamer_smoke()
    print('ALL S4 TESTS PASSED')
