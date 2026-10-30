# S4 设计修订：DeckEncoder（可学习槽位编码 + 双向 Transformer）

> 状态：DESIGNED（S4-lane 激活时的既定设计，首块实现即本模块）。
> 本文取代早期讨论中"deck_ids → 8 嵌入取 mean"的提案。

## 更正声明

早期 S4 提案提出 `deck_ids → 8 个卡嵌入取 mean` 作为 critic 条件向量。
**该提案废弃**：mean 池化丢弃槽位与顺序信息，规划采用基于可学习槽位编码 +
卡牌嵌入的双向 Transformer（下述 DeckEncoder）。

丢弃的三层信息（均为引擎规则/规划要求，非可选）：
1. **槽位身份**：前两个卡槽专属英雄/觉醒卡（引擎规则，card_support
   的 form_availability 按槽位约束）——mean 池化下 critic 无法区分
   "英雄卡在英雄槽"与"任意槽"；
2. **过牌顺序**：deck 顺序决定循环次序（v2 计划原文"牌组序列嵌入，
   顺序=过牌顺序"）；
3. **协同结构**：卡间交互（配合/克制）。

## DeckEncoder 设计

```
输入: deck_ids (..., 8)          # 卡池索引，0 = UNK/空槽
emb  = CardEmbeddingTable(deck_ids)          # (..., 8, d=64)，与 actor 卡头共享
x    = emb + slot_pos[0..7]                  # 可学习槽位编码 (8, d)，广播相加
ctx  = TransformerEncoder(x)                 # 双向 self-attention，1-2 层 × 4 头
cond = mean_over_slots(ctx)                  # (..., d)——attention 后的池化
                                         # （此时每槽已吸收全卡组交互，非盲平均）
```

- **双向而非 causal**：卡组是"带位置的集合"，8 张卡同时在场互相作用，
  没有因果方向可循；双向 self-attention 是该结构的标准解。
- **参数量**：~5-10 万（8 token × 1-2 层 × 4 头），相比 critic 主体
  可忽略；4Hz 决策下计算无感。
- **共享**：卡嵌入与 actor 卡头同表（CardEmbeddingTable）；空槽用
  0 号 UNK 嵌入 + attention mask。
- **纯度合规**：槽位规则是引擎规则（接口）而非人类先验。
- **消费方**：critic 条件向量（S4）；M6 的 V(s0,d) 打分器复用同一
  DeckEncoder；actor 点积卡头不受影响（逐槽打分，槽位身份天然保留）。

## 配套（沿用既有决定）

- CFG dropout p=0.25（序列级）做 V(s,d0) 边缘模式；
- 零和一致性损失与换边消融联动（ablation_paired_swap.md）；
- 覆盖采样器/协同探针、防坍缩监测（支撑集熵、边缘一致性校准）见
  league 设计讨论，S4-lane 激活时一并排期。

## 验收（并入 S4 验收）

- 槽位敏感性：交换 deck 前两槽与普通槽的卡，V 输出应有显著差异
  （mean 池化下恒为零——本设计的直接可测差异）；
- 其余沿用 S4 验收（方差缩减、模式差分、主线不退化）。
