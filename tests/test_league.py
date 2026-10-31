"""League ③④⑤ 单测：门禁、爬山、shadow 流程。"""
import sys, random

sys.path.insert(0, r'D:\pynoob\FirstLight\FirstLight_CR')
sys.path.insert(0, r'D:\pynoob\r2dreamer')

from league import GateChecker, build_pool, hill_climb, DeckSelector


def test_gates():
    g = GateChecker(kl_eps=0.01, kl_windows=3, min_games=10)
    # 门一不足
    ok, r = g.policy_gate([(100, 0.005), (200, 0.006)])
    assert not ok and 'insufficient' in r
    # 门一通过
    ok, r = g.policy_gate([(100, 0.005), (200, 0.006), (300, 0.004)])
    assert ok
    # 门一未平台
    ok, r = g.policy_gate([(100, 0.05), (200, 0.06), (300, 0.04)])
    assert not ok and 'not plateaued' in r
    # 数据门
    stats_empty = []
    ok, r = g.data_gate(stats_empty)
    assert not ok
    stats = [{'opp_deck': list(range(8)), 'win': True, 'reason': 'native_final_result'}] * 12
    ok, r = g.data_gate(stats)
    assert ok
    print('PASS: gates')


def test_hill_climb():
    # 打分函数： Prefer 高数值卡（槽 0 用奇数索引、槽 1 用偶数）
    def score(deck):
        return float(sum(deck) - 2 * sum(1 for i, c in enumerate(deck) if i < 2 and c % 2 == 0))
    rng = random.Random(0)
    candidates = list(range(100, 140))
    start = [101, 102, 103, 104, 105, 106, 107, 108]
    best, best_v, path = hill_climb(start, score, candidates, max_steps=10, rng=rng)
    assert best_v >= score(start)
    assert len(path) >= 1
    # 单调改善
    vals = [p['value'] for p in path]
    assert all(b >= a - 1e-9 for a, b in zip(vals, vals[1:]))
    print(f'PASS: hill_climb (steps={len(path)-1}, v {vals[0]:.1f} -> {vals[-1]:.1f})')


def test_selector_shadow_then_active():
    sel = DeckSelector(epsilon=0.2, min_games=3, max_steps=3, seed=1)
    # 构造 stats：一个对手卡组打了很多局且 agent 全胜（数据门过）
    opp_deck = list(range(9, 17))
    stats = [{'opp_deck': opp_deck, 'win': w, 'reason': 'native_final_result'}
             for w in [True, False] * 6]
    # 门一（KL 探针）不足 → shadow
    out = sel.select(stats, probes=[], wm_plateau=True,
                     score_fn=lambda d: 0.5, agent_deck=list(range(1, 9)))
    assert out['mode'] == 'shadow' and not out['info']['gates_ok']
    # KL 平台 + WM 平台 → active
    probes = [(i * 100, 0.001) for i in range(4)]
    out = sel.select(stats, probes=probes, wm_plateau=True,
                     score_fn=lambda d: float(sum(d)), agent_deck=list(range(1, 9)))
    assert out['mode'] in ('active', 'active_explore')
    assert 'deck0' in out and 'deck1' in out and len(out['deck1']) == 8
    # 爬山路径进 info
    assert 'path' in out['info']
    print(f"PASS: selector shadow→active (mode={out['mode']})")


if __name__ == '__main__':
    test_gates()
    test_hill_climb()
    test_selector_shadow_then_active()
    print('ALL LEAGUE TESTS PASSED')
