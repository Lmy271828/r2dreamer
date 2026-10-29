"""S3 单测：CardCellDist / CardQueryActor / env 技能位编码。"""
import os
import sys

sys.path.insert(0, r'D:\pynoob\FirstLight\FirstLight_CR')
sys.path.insert(0, r'D:\pynoob\r2dreamer')

import numpy as np
import torch

from distributions import CardCellDist
from envs.firstlight import ABILITY_INDEX, ACTION_DIM, CELLS, HAND_NUM

torch.manual_seed(0)


def test_card_cell_dist_masking():
    # 4 手牌中 2 张合法，ABILITY 与 1 张手牌非法；格位只允许 (0,0)
    S, C = 6, 576
    slot_logits = torch.zeros(2, S)
    slot_masks = torch.zeros(2, S)
    slot_masks[:, 0] = 1          # WAIT
    slot_masks[:, 1] = 1          # hand0
    slot_masks[:, 3] = 1          # hand2
    slot_masks[:, 5] = 1          # ABILITY
    cell_logits = torch.zeros(2, S, C)
    cell_masks = torch.zeros(2, S, C)
    cell_masks[:, 1, 0] = 1       # hand0 仅格 0
    cell_masks[:, 3, 5] = 1       # hand2 仅格 5
    cell_masks[:, 0, :] = 1       # WAIT/ABILITY 占位全 1
    cell_masks[:, 5, :] = 1

    dist = CardCellDist(slot_logits, slot_masks, cell_logits, cell_masks, unimix_ratio=0.0)

    # 大样本下非法槽位从不被采样
    slots, cells = dist.rsample()
    for _ in range(200):
        s, c = dist.rsample()
        slots = torch.cat([slots, s]); cells = torch.cat([cells, c])
    si = slots.argmax(-1)
    assert not torch.isin(si, torch.tensor([2, 4])).any(), "illegal slot sampled"

    # log_prob 与手算一致：p(slot3)*p(cell5|slot3)
    val = (torch.zeros(1, S).index_fill_(1, torch.tensor([3]), 1.0),
           torch.zeros(1, C).index_fill_(1, torch.tensor([5]), 1.0))
    lp = dist.log_prob(val)
    p_slot = torch.softmax(dist.slot_logprobs, -1)[0, 3]
    p_cell = torch.exp(dist.cell_logprobs)[0, 3, 5]
    assert torch.allclose(lp, torch.log(p_slot * p_cell), atol=1e-5), (lp, p_slot, p_cell)

    # 非法槽位 log_prob 远小于合法槽位（-1e9 洪泛 + softmax 归一）
    val_bad = (torch.zeros(1, S).index_fill_(1, torch.tensor([2]), 1.0),
               torch.zeros(1, C))
    assert dist.log_prob(val_bad) < dist.log_prob(val) - 10.0

    # entropy = H(slot) + Σ p(s) H(cell|s)；手算对照（entr 约定 0·(-inf)=0）
    h_slot = torch.special.entr(torch.exp(dist.slot_logprobs)).sum(-1)
    h_cell = torch.special.entr(torch.exp(dist.cell_logprobs)).sum(-1)
    sp = torch.exp(dist.slot_logprobs)
    expect = (h_slot + (sp * h_cell).sum(-1))
    assert torch.allclose(dist.entropy(), expect, atol=1e-4) and torch.isfinite(dist.entropy()).all()

    # mode：合法槽位优先 + 该槽位最优格
    slot_m, cell_m = dist.mode()
    assert slot_m[0].argmax() == 0  # logits 全零 → argmax 先 WAIT
    assert cell_m[0].argmax() == 0
    print("PASS: CardCellDist masking/log_prob/entropy/mode")


def test_card_query_actor():
    from envs.firstlight import DECK_INDEX, DECK_POOL
    from networks import CardEmbeddingTable, CardQueryActor

    class Cfg:  # MLP(config, ...) 读顶层字段
        name = "cell_mlp"; layers = 2; units = 64; act = "ELU"
        symlog_inputs = False; outscale = 1.0; device = "cpu"

    vocab = len(DECK_POOL) + 1
    table = CardEmbeddingTable(vocab, dim=32)
    actor = CardQueryActor(Cfg, inp_dim=128, card_table=table, cell_num=CELLS, unimix_ratio=0.0)
    feat = torch.randn(2, 3, 128)
    hand_ids = torch.randint(1, vocab, (2, 3, HAND_NUM))
    mask = torch.zeros(2, 3, ACTION_DIM)
    mask[..., 0] = 1                                   # WAIT
    mask[..., 1:1 + CELLS] = 1                         # hand0 全格合法
    mask[..., 1 + 2 * CELLS] = 1                       # hand2 一格
    mask[..., ABILITY_INDEX] = 1                       # 技能位放开

    dist = actor(feat, hand_ids, mask)
    assert dist.S == 6 and dist.C == CELLS
    slot, cell = dist.rsample()
    assert slot.shape == (2, 3, 6) and cell.shape == (2, 3, CELLS)
    si = slot.argmax(-1)
    assert not torch.isin(si, torch.tensor([2, 4])).any(), "illegal hand slot sampled"
    # 技能位能被采样到（掩码放开 + 足够多样本）
    seen = set()
    for _ in range(300):
        s, _ = dist.rsample()
        seen |= set(s.argmax(-1).flatten().tolist())
    assert 5 in seen, "ability slot never sampled though masked legal"
    # 掩码关闭技能位后不再采样
    mask[..., ABILITY_INDEX] = 0
    dist2 = actor(feat, hand_ids, mask)
    seen2 = set()
    for _ in range(200):
        s, _ = dist2.rsample()
        seen2 |= set(s.argmax(-1).flatten().tolist())
    assert 5 not in seen2, "ability sampled while masked off"
    # log_prob 可反传
    lp = dist.log_prob(dist.rsample())
    lp.mean().backward()
    grads = [p.grad for p in actor.parameters() if p.grad is not None]
    assert len(grads) > 5, "no gradient flow"
    print("PASS: CardQueryActor dynamic candidates + ability token + grad flow")


def test_env_action_contract():
    # 常量与扁平掩码编码约定
    assert ABILITY_INDEX == 1 + HAND_NUM * CELLS == 2305
    assert ACTION_DIM == 2306

    from envs.firstlight import FirstLight as E
    env = object.__new__(E)
    class Mask:
        hand_slots = [True, True, False, False]
        placement_masks = {"0": {"row_major": [[False] * 18 for _ in range(32)]},
                           "1": {"row_major": [[True] * 18 for _ in range(32)]}}
        ability_sources = (101, 202)
    env._mask = Mask()
    flat = env._mask_flat()
    assert flat.shape[0] == ACTION_DIM and flat[ABILITY_INDEX] == 1
    assert flat[1 + 2 * CELLS] == 0 and flat[1 + CELLS] == 1  # hand2 非法/hand1 合法
    # decode 哨兵与合法判定（仍有就绪技能时）
    assert env._decode(ABILITY_INDEX) == (-1, None, None)
    assert env._legal(-1, None, None) is True
    Mask.ability_sources = ()
    assert env._mask_flat()[ABILITY_INDEX] == 0
    assert env._legal(-1, None, None) is False
    # 实体解析
    env._ability_by_entity = {202: "Deflect", 101: "ChampGuardianAbility"}
    Mask.ability_sources = (101, 202)
    assert env._resolve_ability() == (101, "ChampGuardianAbility")  # canonical 升序取最小
    env._ability_by_entity = {202: "Deflect"}
    assert env._resolve_ability() == (202, "Deflect")
    env._ability_by_entity = {}
    assert env._resolve_ability() is None
    print("PASS: env action contract 2306/ability bit/decode/resolve")


if __name__ == "__main__":
    test_card_cell_dist_masking()
    test_card_query_actor()
    test_env_action_contract()
    print("ALL S3 TESTS PASSED")
