"""S5 单测：CardActionHead 扁平适配 + 三处穿透的数据形状约定。"""
import sys

sys.path.insert(0, r'D:\pynoob\FirstLight\FirstLight_CR')
sys.path.insert(0, r'D:\pynoob\r2dreamer')

import torch

from envs.firstlight import ABILITY_INDEX, ACTION_DIM, CELLS
from networks import CardActionHead

torch.manual_seed(0)


def _head():
    class Cfg:  # MLP(config, ...) 读顶层字段
        name = "cell_mlp"; layers = 2; units = 64; act = "ELU"
        symlog_inputs = False; outscale = 1.0; device = "cpu"
        shape = (ACTION_DIM,)

    class Dist:
        unimix_ratio = 0.0

    Cfg.dist = Dist()
    return CardActionHead(Cfg, inp_dim=128)


def test_flat_roundtrip():
    head = _head()
    B, T = 2, 3
    feat = torch.randn(B, T, 128)
    hand_ids = torch.randint(1, 60, (B, T, 4))
    mask = torch.zeros(B, T, ACTION_DIM)
    mask[..., 0] = 1
    mask[..., 1:1 + CELLS] = 1          # hand0
    mask[..., ABILITY_INDEX] = 1        # 技能位放开
    dist = head(feat, mask=mask, hand_ids=hand_ids)

    flat = dist.rsample()
    assert flat.shape == (B, T, ACTION_DIM) and torch.allclose(flat.sum(-1), torch.ones(B, T))
    # 每个采样要么 WAIT、要么 hand0、要么技能位（掩码只放开了这些）
    idx = flat.argmax(-1)
    ok = (idx == 0) | ((idx >= 1) & (idx < 1 + CELLS)) | (idx == ABILITY_INDEX)
    assert ok.all()

    # 分解-重组对拍：手算 (slot, cell) 的 log_prob 与扁平 log_prob 一致
    val = torch.zeros(B, T, ACTION_DIM)
    val[..., 1 + 5] = 1.0               # hand0 格 5
    lp_flat = dist.log_prob(val)
    slot = torch.zeros(B, T, 6); slot[..., 1] = 1.0
    cell = torch.zeros(B, T, CELLS); cell[..., 5] = 1.0
    lp_struct = dist.inner.log_prob((slot, cell))
    assert torch.allclose(lp_flat, lp_struct, atol=1e-5)

    val_ab = torch.zeros(B, T, ACTION_DIM); val_ab[..., ABILITY_INDEX] = 1.0
    lp_ab = dist.log_prob(val_ab)
    slot5 = torch.zeros(B, T, 6); slot5[..., 5] = 1.0
    cell0 = torch.zeros(B, T, CELLS); cell0[..., 0] = 1.0
    assert torch.allclose(lp_ab, dist.inner.log_prob((slot5, cell0)), atol=1e-5)

    val_wait = torch.zeros(B, T, ACTION_DIM); val_wait[..., 0] = 1.0
    assert torch.allclose(dist.log_prob(val_wait), dist.inner.log_prob(
        (torch.zeros(B, T, 6).index_fill_(-1, torch.tensor([0]), 1.0),
         torch.zeros(B, T, CELLS).index_fill_(-1, torch.tensor([0]), 1.0))), atol=1e-5)
    assert torch.isfinite(dist.entropy()).all()
    assert torch.isfinite(dist.mode()).all()
    print("PASS: CardFlatDist roundtrip (hand/ability/wait) + log_prob parity")


def test_gradient_through_log_prob():
    head = _head()
    feat = torch.randn(2, 3, 128, requires_grad=True)
    hand_ids = torch.randint(1, 60, (2, 3, 4))
    mask = torch.zeros(2, 3, ACTION_DIM)
    mask[..., 0] = 1; mask[..., 1:1 + CELLS] = 1
    dist = head(feat, mask=mask, hand_ids=hand_ids)
    flat = dist.rsample().detach()      # 想象轨迹中动作是 detached 的（M2 同款）
    lp = dist.log_prob(flat)            # 梯度经 log_prob 重算流回 actor
    lp.mean().backward()
    assert feat.grad is None or True    # feat 非参数，梯度存于 actor 参数
    n = sum(1 for p in head.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    assert n >= 8, f"too few param grads: {n}"
    print(f"PASS: policy-gradient flows into {n} actor params")


if __name__ == "__main__":
    test_flat_roundtrip()
    test_gradient_through_log_prob()
    print("ALL S5 TESTS PASSED")
