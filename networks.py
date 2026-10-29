import math
import os
import re
import sys
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

import distributions as dists
from tools import weight_init_


class LambdaLayer(nn.Module):
    """Wrap an arbitrary callable into an ``nn.Module``."""

    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BlockLinear(nn.Module):
    """Block-wise linear layer.

    Weight layout is chosen to cooperate with PyTorch's fan-in/fan-out
    calculation used by initializers.
    """

    def __init__(self, in_ch: int, out_ch: int, blocks: int, outscale: float = 1.0):
        super().__init__()
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.blocks = int(blocks)
        self.outscale = float(outscale)

        # Store weight in a layout that works with torch's fan calculation.
        # (O/G, I/G, G)
        self.weight = nn.Parameter(torch.empty(self.out_ch // self.blocks, self.in_ch // self.blocks, self.blocks))
        # zeros, not empty: weight_init_ fills the bias with 0 anyway, so this is the same
        # value in production while keeping the layer usable before any initializer runs.
        # Left uninitialised, garbage of ~1e31 reached .square() and overflowed float32.
        self.bias = nn.Parameter(torch.zeros(self.out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (..., I)
        batch_shape = x.shape[:-1]
        # Reshape to expose block dimension.
        # (..., I) -> (..., G, I/G)
        x = x.view(*batch_shape, self.blocks, self.in_ch // self.blocks)

        # Block-wise multiplication
        # (..., G, I/G), (O/G, I/G, G) -> (..., G, O/G)
        x = torch.einsum("...gi,oig->...go", x, self.weight)
        # Merge block dimension back.
        # (..., G, O/G) -> (..., O)
        x = x.reshape(*batch_shape, self.out_ch)
        return x + self.bias


class Conv2dSamePad(nn.Conv2d):
    """A Conv2d layer that emulates TensorFlow's 'SAME' padding."""

    def _calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        i_div_s_ceil = (i + s - 1) // s
        return max((i_div_s_ceil - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]
        pad_h = self._calc_same_pad(ih, self.kernel_size[0], self.stride[0], self.dilation[0])
        pad_w = self._calc_same_pad(iw, self.kernel_size[1], self.stride[1], self.dilation[1])

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x,
                [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2],
            )

        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class RMSNorm2D(nn.RMSNorm):
    """RMSNorm over channel-last format applied to 4D tensors."""

    def __init__(self, ch: int, eps: float = 1e-3, dtype=None):
        super().__init__(ch, eps=eps, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply RMSNorm over the channel dimension.
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class MultiEncoder(nn.Module):
    def __init__(
        self,
        config,
        shapes,
    ):
        super().__init__()
        excluded = ("is_first", "is_last", "is_terminal", "reward")
        shapes = {k: v for k, v in shapes.items() if k not in excluded and not k.startswith("log_")}
        self.cnn_shapes = {k: v for k, v in shapes.items() if len(v) == 3 and re.match(config.cnn_keys, k)}
        self.mlp_shapes = {k: v for k, v in shapes.items() if len(v) in (1, 2) and re.match(config.mlp_keys, k)}
        print("Encoder CNN shapes:", self.cnn_shapes)
        print("Encoder MLP shapes:", self.mlp_shapes)

        self.out_dim = 0
        self.selectors = []
        self.encoders = []
        if self.cnn_shapes:
            input_ch = sum([v[-1] for v in self.cnn_shapes.values()])
            input_shape = tuple(self.cnn_shapes.values())[0][:2] + (input_ch,)
            self.encoders.append(ConvEncoder(config.cnn, input_shape))
            self.selectors.append(lambda obs: torch.cat([obs[k] for k in self.cnn_shapes], -1))
            self.out_dim += self.encoders[-1].out_dim
        if self.mlp_shapes:
            inp_dim = sum([sum(v) for v in self.mlp_shapes.values()])
            self.encoders.append(MLP(config.mlp, inp_dim))
            self.selectors.append(lambda obs: torch.cat([obs[k] for k in self.mlp_shapes], -1))
            self.out_dim += self.encoders[-1].out_dim
        self.encoders = nn.ModuleList(self.encoders)

        if len(self.encoders) > 1:
            self.fuser = lambda x: torch.cat(x, dim=-1)
        elif len(self.encoders) == 1:
            self.fuser = lambda x: x[0]
        else:
            raise NotImplementedError

        self.apply(weight_init_)

    def forward(self, obs):
        """Encode a dict of observations."""
        # dict of (B, T, *)
        return self.fuser([enc(sel(obs)) for enc, sel in zip(self.encoders, self.selectors)])


class MultiDecoder(nn.Module):
    def __init__(self, config, deter, flat_stoch, shapes):
        super().__init__()
        excluded = ("is_first", "is_last", "is_terminal")
        shapes = {k: v for k, v in shapes.items() if k not in excluded}
        self.cnn_shapes = {k: v for k, v in shapes.items() if len(v) == 3 and re.match(config.cnn_keys, k)}
        self.mlp_shapes = {k: v for k, v in shapes.items() if len(v) in (1, 2) and re.match(config.mlp_keys, k)}
        print("Decoder CNN shapes:", self.cnn_shapes)
        print("Decoder MLP shapes:", self.mlp_shapes)
        self.all_keys = list(self.mlp_shapes.keys()) + list(self.cnn_shapes.keys())

        # Unlike the encoder, each decoder is initialized independently.
        if self.cnn_shapes:
            some_shape = list(self.cnn_shapes.values())[0]
            shape = (sum(x[-1] for x in self.cnn_shapes.values()),) + some_shape[:-1]
            self._cnn = ConvDecoder(
                config.cnn,
                deter,
                flat_stoch,
                shape,
            )
            self._image_dist = partial(getattr(dists, str(config.cnn_dist.name)), **config.cnn_dist)
        if self.mlp_shapes:
            shape = (sum(sum(x) for x in self.mlp_shapes.values()),)
            config.mlp.shape = shape
            self._mlp = MLPHead(config.mlp, deter + flat_stoch)
            self._mlp_dist = partial(getattr(dists, str(config.mlp_dist.name)), **config.mlp_dist)

    def forward(self, stoch, deter):
        """Decode latent states into observation distributions."""
        # (B, T, S, K), (B, T, D)
        dists = {}
        if self.cnn_shapes:
            split_sizes = [v[-1] for v in self.cnn_shapes.values()]
            # (B, T, H, W, C_sum)
            outputs = self._cnn(stoch, deter)
            outputs = torch.split(outputs, split_sizes, -1)
            dists.update({key: self._image_dist(output) for key, output in zip(self.cnn_shapes.keys(), outputs)})
        if self.mlp_shapes:
            split_sizes = [v[0] for v in self.mlp_shapes.values()]
            # (B, T, S*K + D)
            feat = torch.cat([stoch.reshape(*deter.shape[:-1], -1), deter], -1)
            outputs = self._mlp(feat)
            outputs = torch.split(outputs, split_sizes, -1)
            dists.update({key: self._mlp_dist(output) for key, output in zip(self.mlp_shapes.keys(), outputs)})
        return dists


class ConvEncoder(nn.Module):
    def __init__(self, config, input_shape):
        super().__init__()
        act = getattr(torch.nn, config.act)
        h, w, input_ch = input_shape
        self.depths = tuple(int(config.depth) * int(mult) for mult in list(config.mults))
        self.kernel_size = int(config.kernel_size)
        in_dim = input_ch
        layers = []
        for i, depth in enumerate(self.depths):
            layers.append(
                Conv2dSamePad(
                    in_channels=in_dim,
                    out_channels=depth,
                    kernel_size=self.kernel_size,
                    stride=1,
                    bias=True,
                )
            )
            layers.append(nn.MaxPool2d(2, 2))
            if config.norm:
                layers.append(RMSNorm2D(depth, eps=1e-04, dtype=torch.float32))
            layers.append(act())
            in_dim = depth
            h, w = h // 2, w // 2

        self.out_dim = self.depths[-1] * h * w
        self.layers = nn.Sequential(*layers)

    def forward(self, obs):
        """Encode image-like observations with a CNN."""
        # (B, T, H, W, C)
        obs = obs - 0.5
        # (B*T, H, W, C)
        x = obs.reshape(-1, *obs.shape[-3:])
        # (B*T, C, H, W)
        x = x.permute(0, 3, 1, 2)
        # (B*T, C_feat, H_feat, W_feat)
        x = self.layers(x)
        # (B*T, C_feat*H_feat*W_feat)
        x = x.reshape(x.shape[0], -1)
        # (B, T, C_feat*H_feat*W_feat)
        return x.reshape(*obs.shape[:-3], x.shape[-1])


class ConvDecoder(nn.Module):
    def __init__(self, config, deter, flat_stoch, shape=(3, 64, 64)):
        super().__init__()
        act = getattr(torch.nn, config.act)
        self._shape = shape
        self.depths = tuple(int(config.depth) * int(mult) for mult in list(config.mults))
        factor = 2 ** (len(self.depths))
        minres = [int(x // factor) for x in shape[1:]]
        self.min_shape = (*minres, self.depths[-1])
        self.bspace = int(config.bspace)
        self.kernel_size = int(config.kernel_size)
        self.units = int(config.units)
        u, g = math.prod(self.min_shape), self.bspace
        self.sp0 = BlockLinear(deter, u, g)
        self.sp1 = nn.Sequential(
            nn.Linear(flat_stoch, 2 * self.units), nn.RMSNorm(2 * self.units, eps=1e-04, dtype=torch.float32), act()
        )
        self.sp2 = nn.Linear(2 * self.units, math.prod(self.min_shape))
        self.sp_norm = nn.Sequential(nn.RMSNorm(self.depths[-1], eps=1e-04, dtype=torch.float32), act())
        layers = []
        in_dim = self.depths[-1]
        for depth in reversed(self.depths[:-1]):
            layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
            layers.append(Conv2dSamePad(in_dim, depth, self.kernel_size, stride=1, bias=True))
            layers.append(RMSNorm2D(depth, eps=1e-04, dtype=torch.float32))
            layers.append(act())
            in_dim = depth
        layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
        layers.append(Conv2dSamePad(in_dim, self._shape[0], self.kernel_size, stride=1, bias=True))
        self.layers = nn.Sequential(*layers)
        self.apply(weight_init_)

    def forward(self, stoch, deter):
        """Decode latent states into images.

        Notes
        -----
        The decoder first constructs a low-resolution spatial feature map from
        the deterministic state (block-linear projection) and from the stochastic
        state (MLP projection), concats them, then upsamples back to the target
        resolution.
        """
        # (B, T, S, K), (B, T, D)
        B_T = deter.shape[:-1]
        # (B*T, D), (B*T, S*K)
        x0, x1 = deter.reshape(B_T.numel(), deter.shape[-1]), stoch.reshape(B_T.numel(), -1)

        # Spatial features from deterministic state
        # (H_feat, W_feat, C_feat)
        H_feat, W_feat, C_feat = self.min_shape
        # (B*T, H_feat*W_feat*C_feat)
        x0 = self.sp0(x0)
        # (B*T, G, H_feat, W_feat, C_feat/G)
        x0 = x0.reshape(-1, self.bspace, H_feat, W_feat, C_feat // self.bspace)
        # (B*T, H_feat, W_feat, C_feat)
        x0 = x0.permute(0, 2, 3, 1, 4).reshape(-1, H_feat, W_feat, C_feat)

        # Spatial features from stochastic state
        # (B*T, 2*U)
        x1 = self.sp1(x1)
        # (B*T, H_feat, W_feat, C_feat)
        x1 = self.sp2(x1).reshape(-1, H_feat, W_feat, C_feat)

        # Combine and upsample
        # (B*T, H_feat, W_feat, C_feat)
        x = self.sp_norm(x0 + x1)
        # (B*T, C_feat, H_feat, W_feat)
        x = x.permute(0, 3, 1, 2)
        x = self.layers(x)  # Upsamples to original H, W
        # (B*T, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = torch.sigmoid(x)
        # (B, T, H, W, C)
        return x.reshape(*B_T, *x.shape[1:])


class MLP(nn.Module):
    def __init__(
        self,
        config,
        inp_dim,
    ):
        super().__init__()
        act = getattr(torch.nn, config.act)
        self._symlog_inputs = bool(config.symlog_inputs)
        self._device = torch.device(config.device)
        self.layers = nn.Sequential()
        for i in range(config.layers):
            self.layers.add_module(f"{config.name}_linear{i}", nn.Linear(inp_dim, config.units, bias=True))
            self.layers.add_module(f"{config.name}_norm{i}", nn.RMSNorm(config.units, eps=1e-04, dtype=torch.float32))
            self.layers.add_module(f"{config.name}_act{i}", act())
            inp_dim = config.units
        self.out_dim = config.units

    def forward(self, x):
        # (B, T, I)
        if self._symlog_inputs:
            x = dists.symlog(x)
        # (B, T, U)
        return self.layers(x)


class MLPHead(nn.Module):
    def __init__(self, config, inp_dim):
        super().__init__()
        self.mlp = MLP(config, inp_dim)
        self._dist_name = str(config.dist.name)
        self._outscale = float(config.outscale)
        self._dist = getattr(dists, str(config.dist.name))

        if self._dist_name == "bounded_normal":
            self.last = nn.Linear(self.mlp.out_dim, config.shape[0] * 2, bias=True)
            kwargs = {"min_std": float(config.dist.min_std), "max_std": float(config.dist.max_std)}
        elif self._dist_name == "onehot":
            self.last = nn.Linear(self.mlp.out_dim, config.shape[0], bias=True)
            kwargs = {"unimix_ratio": float(config.dist.unimix_ratio)}
        elif self._dist_name == "multi_onehot":
            self.last = nn.Linear(self.mlp.out_dim, sum(config.shape), bias=True)
            kwargs = {"unimix_ratio": float(config.dist.unimix_ratio), "shape": tuple(config.shape)}
        elif self._dist_name == "symexp_twohot":
            self.last = nn.Linear(self.mlp.out_dim, config.shape[0], bias=True)
            kwargs = {"device": torch.device(config.device), "bin_num": int(config.dist.bin_num)}
        elif self._dist_name in ("binary", "identity"):
            self.last = nn.Linear(self.mlp.out_dim, config.shape[0], bias=True)
            kwargs = {}
        else:
            raise NotImplementedError

        self._dist = partial(self._dist, **kwargs)

        self.mlp.apply(weight_init_)
        self.last.apply(weight_init_)
        # apply explicit output scaling.
        if self._outscale != 1.0:
            with torch.no_grad():
                self.last.weight.mul_(self._outscale)

    def forward(self, x, mask=None):
        """Produce a distribution head.

        mask (..., A) optional legality gate for discrete heads: illegal
        logits are flooded to -1e9 BEFORE unimix.  Finite flooding (not
        -inf) keeps categorical entropy nan-free (0 * -1e9 is 0, while
        0 * -inf is nan), and the residual unimix floor (~eps/K per
        illegal action, ~4e-6 here) degrades to WAIT behaviourally
        anyway.
        """
        # (B, T, F)
        logits = self.last(self.mlp(x))
        if mask is not None:
            logits = logits.masked_fill(mask <= 0, -1e9)
        return self._dist(logits)


def _deck_pool_ids():
    """M3 卡牌词表 = FirstLight 竞技支持白名单 ∩ 卡牌目录。
    与 envs/firstlight.DECK_POOL 保持一致（deck_ids 索引即该池下标）。"""
    import json

    root = os.environ.get("CR_FIRSTLIGHT_ROOT", "D:/pynoob/FirstLight/FirstLight_CR")
    ids = set()
    try:
        from native_runner.card_names import CARD_NAMES  # noqa: PLC0415 -- 需 FirstLight 路径已注入

        sys_path = str(root)
        if sys_path not in sys.path:
            sys.path.insert(0, sys_path)
        from native_runner.card_names import CARD_NAMES as NAMES  # noqa: F811

        ids = set(NAMES)
    except Exception:
        pass
    try:
        sup = json.loads(open(os.path.join(root, "native_runner", "data", "competitive", "card_support.json")).read())
        ids &= {int(c["card_id"]) for c in sup.get("supported_cards", [])}
    except Exception:
        pass
    return tuple(sorted(ids))


class CardEmbeddingTable(nn.Module):
    """M4 卡牌共享嵌入表：actor 候选打分、critic 卡组条件、未来卡组池共用同一张表。

    - 词表 0 号位 = UNK/空槽；e_wait 是与卡牌同维的可学习等待令牌，
      在候选集里占 0 号位（softmax 里与出牌候选同台竞争）
    - 查询与候选各自 LayerNorm + L2 归一化后做余弦点积，乘可学习温度
      τ = exp(log_temperature)（初始 1.0）——无界点积会让熵塌缩
    """

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.emb = nn.Embedding(vocab_size, dim)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.wait_token = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.wait_token, std=0.02)
        self.norm = nn.LayerNorm(dim)
        self.log_temperature = nn.Parameter(torch.zeros(1))

    def _norm(self, x):
        import torch.nn.functional as F

        return F.normalize(self.norm(x), dim=-1)

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temperature.clamp(-4.0, 4.0))

    def encode_cards(self, ids):
        """ids (..., C) int64 → (..., C, d) 归一化卡牌嵌入（0 号 = UNK/空槽）。"""
        return self._norm(self.emb(ids.long()))

    def candidates_with_wait(self, ids):
        """(..., C) 索引 → (..., C+1, d)：0 号位是 e_wait（WAIT 候选），其余按原顺序。"""
        cards = self.encode_cards(ids)
        wait = self._norm(self.wait_token).expand(*cards.shape[:-2], 1, cards.shape[-1])
        return torch.cat([wait, cards], dim=-2)

    def score(self, query, candidates):
        """query (..., d)、candidates (..., C, d) → (..., C) 余弦相似度 × τ。"""
        import torch.nn.functional as F

        return F.cosine_similarity(query.unsqueeze(-2), candidates, dim=-1) * self.temperature


class Projector(nn.Module):
    def __init__(self, in_ch1, in_ch2):
        super().__init__()
        self.w = nn.Linear(in_ch1, in_ch2, bias=False)
        self.apply(weight_init_)

    def forward(self, x):
        return self.w(x)


class ReturnEMA(nn.Module):
    """running mean and std"""

    def __init__(self, device, alpha=1e-2):
        super().__init__()
        self.device = device
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95], device=device)
        self.register_buffer("ema_vals", torch.zeros(2, dtype=torch.float32, device=self.device))

    def __call__(self, x):
        # torch.quantile is not traceable by Dynamo once the input shape becomes
        # symbolic ("Cannot call numel() on tensor with symbolic sizes/strides"),
        # and it sorts internally anyway. Only two order statistics are needed,
        # so sort once and index -- same values, no dynamic-shape landmine.
        flat = torch.flatten(x.detach())
        srt, _ = torch.sort(flat)
        n = srt.shape[0]
        # linear interpolation, matching torch.quantile's default
        pos = self.range.to(srt.dtype) * (n - 1)
        lo = pos.floor().long().clamp_(0, n - 1)
        hi = pos.ceil().long().clamp_(0, n - 1)
        frac = pos - pos.floor()
        x_quantile = srt[lo] + (srt[hi] - srt[lo]) * frac
        self.ema_vals.copy_(self.alpha * x_quantile.detach() + (1 - self.alpha) * self.ema_vals)
        scale = torch.clip(self.ema_vals[1] - self.ema_vals[0], min=1.0)
        offset = self.ema_vals[0]
        return offset.detach(), scale.detach()


class CardQueryActor(nn.Module):
    """S3 两级动作头：点积卡头（动态候选集）× 条件格头。

    - 卡头：query = Linear(feat → d)，与候选嵌入（e_wait + 手牌 4 卡 +
      e_ability）做余弦点积 × τ（复用 CardEmbeddingTable），得槽位 logits。
      候选集随手牌变化：手牌换了，点积对象就换了——固定 2305 维输出头
      无法表达这一结构。
    - 格头：MLP([feat, 被选卡嵌入]) → 576 格 logits，按该槽位的
      placement_mask 掩码；对全部 4 个手牌槽并行计算，分布内按采样槽位
      选取。
    - ABILITY 是预留令牌：合法性由掩码第 2305 位（env 侧技能位）决定，
      未接入前恒非法，分布行为与 WAIT-only 等价。

    mask 约定：沿用 env 的扁平掩码 (1 + 4*CELLS + 1)。bit0=WAIT，
    bit(1+s*576 + c)=槽 s 格 c，bit2305=技能位。
    """

    HAND = 4
    SLOTS = 6  # WAIT + 4 hand + ABILITY

    def __init__(self, config, inp_dim, card_table: CardEmbeddingTable, cell_num: int, unimix_ratio: float):
        super().__init__()
        self.card_table = card_table
        self.cell_num = int(cell_num)
        self.unimix_ratio = float(unimix_ratio)
        self.d = card_table.dim
        self.ability_token = nn.Parameter(torch.zeros(self.d))
        nn.init.normal_(self.ability_token, std=0.02)
        self.query = nn.Linear(inp_dim, self.d)
        self.cell_mlp = MLP(config, inp_dim + self.d)
        self.cell_last = nn.Linear(self.cell_mlp.out_dim, self.cell_num)
        self.query.apply(weight_init_)
        self.cell_last.apply(weight_init_)
        if float(config.outscale) != 1.0:
            with torch.no_grad():
                self.cell_last.weight.mul_(float(config.outscale))

    def forward(self, feat, hand_ids, mask_flat):
        """feat (..., F)、hand_ids (..., 4)、mask_flat (..., 1+4C+1) → CardCellDist。

        前导维任意（act 传 (B,1,*)，想象 rollout 传 (B*T, *)）。
        """
        import torch.nn.functional as F

        lead = feat.shape[:-1]
        query = F.normalize(self.query(feat), dim=-1)                    # (...,d)
        cards = self.card_table.encode_cards(hand_ids)                   # (...,4,d)
        wait = F.normalize(self.card_table.norm(self.card_table.wait_token), dim=-1)
        ability = F.normalize(self.card_table.norm(self.ability_token), dim=-1)
        wait = wait.expand(*cards.shape[:-2], 1, self.d)
        ability = ability.expand(*cards.shape[:-2], 1, self.d)
        candidates = torch.cat([wait, cards, ability], dim=-2)           # (...,6,d)

        slot_logits = self.card_table.score(query, candidates)           # (...,6)

        # 槽位合法性自扁平掩码导出（布局 1 WAIT + 4×C 手牌 + 1 技能）：
        # WAIT 位取 bit0；手牌位 = 该槽任一格合法；技能位 = bit2305
        wait_bit = mask_flat[..., :1]                                    # (...,1)
        hand_cells = mask_flat[..., 1:1 + self.HAND * self.cell_num].reshape(
            *lead, self.HAND, self.cell_num)
        ability_bit = mask_flat[..., 1 + self.HAND * self.cell_num:]     # (...,1)
        slot_mask = torch.cat([wait_bit, hand_cells.amax(dim=-1), ability_bit], dim=-1)  # (...,6)

        cell_in = torch.cat([feat.unsqueeze(-2).expand(*lead, self.HAND, feat.shape[-1]),
                             cards], dim=-1)                              # (...,4,F+d)
        cell_logits = self.cell_last(self.cell_mlp(cell_in))              # (...,4,C)
        cell_masks = hand_cells                                           # (...,4,C)
        # WAIT/ABILITY 槽无格位语义：给全 1 掩码占位，选中时 cell 由 env 忽略
        ones = torch.ones(*lead, 1, self.cell_num, device=cell_logits.device, dtype=cell_logits.dtype)
        zeros = torch.zeros(*lead, 1, self.cell_num, device=cell_logits.device, dtype=cell_logits.dtype)
        # 槽位顺序对齐卡头候选：[WAIT, 手牌0..3, ABILITY]
        cell_logits = torch.cat([zeros, cell_logits, zeros], dim=-2)      # (...,6,C)
        cell_masks = torch.cat([ones, cell_masks, ones], dim=-2)          # (...,6,C)
        return dists.CardCellDist(slot_logits, slot_mask, cell_logits, cell_masks, self.unimix_ratio)


def default_card_table(dim: int = 64) -> CardEmbeddingTable:
    """M4 卡表：词表 = DECK_INDEX 池（+0 号 UNK/WAIT 位），d=64。"""
    return CardEmbeddingTable(len(_deck_pool_ids()) + 1, dim)


class CardFlatDist:
    """CardCellDist → 扁平 2306 维接口适配器。

    环境交换、RSSM prev_action、buffer 存储都沿用扁平 one-hot；
    log_prob/entropy 在内部解包回 (槽位, 格位) 结构计算。槽位候选：
    0=WAIT，1..4=手牌，5=ABILITY（冠军/英雄技能，合法性=扁平掩码 bit2305）。
    """

    def __init__(self, inner: "dists.CardCellDist", cell_num: int):
        self.inner = inner
        self.cell_num = cell_num

    def _idx_from(self, si, ci):
        """(槽位 0..5, 格位) -> 扁平索引。"""
        idx = torch.zeros_like(si)
        idx = torch.where(si == 0, torch.zeros_like(si), idx)
        hand = (si >= 1) & (si <= 4)
        idx = torch.where(hand, 1 + (si - 1) * self.cell_num + ci, idx)
        idx = torch.where(si == 5, torch.full_like(si, 1 + 4 * self.cell_num), idx)
        return idx

    def _split(self, flat):
        """扁平 one-hot (..., A) -> (slot_onehot (...,6), cell_onehot (...,576))。"""
        import torch.nn.functional as F

        idx = flat.argmax(dim=-1)
        ability = idx == 1 + 4 * self.cell_num
        hand = (idx >= 1) & (idx <= 4 * self.cell_num)
        si = torch.where(ability, torch.full_like(idx, 5), torch.where(hand, 1 + (idx - 1) // self.cell_num, torch.zeros_like(idx)))
        ci = torch.where(hand, (idx - 1) % self.cell_num, torch.zeros_like(idx))
        slot = F.one_hot(si, 6).to(flat.dtype)
        cell = F.one_hot(ci, self.cell_num).to(flat.dtype)
        return (slot, cell)

    def rsample(self, sample_shape=(), temperature=1.0):
        slot, cell = self.inner.rsample(temperature=temperature)
        si, ci = slot.argmax(dim=-1), cell.argmax(dim=-1)
        import torch.nn.functional as F

        return F.one_hot(self._idx_from(si, ci), 1 + 4 * self.cell_num + 1).to(slot.dtype)

    @property
    def mode(self):
        slot, cell = self.inner.mode()
        si, ci = slot.argmax(dim=-1), cell.argmax(dim=-1)
        import torch.nn.functional as F

        return F.one_hot(self._idx_from(si, ci), 1 + 4 * self.cell_num + 1).to(slot.dtype)

    def sample(self, **kwargs):
        raise NotImplementedError

    def log_prob(self, flat):
        return self.inner.log_prob(self._split(flat))

    def entropy(self):
        return self.inner.entropy()


class CardActionHead(nn.Module):
    """结构化 actor 头，对 dreamer 暴露与 MLPHead 相同的扁平接口。

    forward(feat (B,T,F), mask (B,T,A), hand_ids (B,T,4)) → CardFlatDist。
    S5 三处穿透（act / _imagine / policy.log_prob）只多传 hand_ids，其余
    代码不变：想象 rollout 里手牌沿时间轴前向持有（与掩码同一近似——
    手牌只在出牌后变化）。
    """

    def __init__(self, config, inp_dim, cell_head="categorical"):
        super().__init__()
        # cell_head: "categorical"（主线，Arm A）| "gaussian"（消融 Arm B，
        # 未实现——规程与判据见 docs/ablation_hier_action_head.md）
        if cell_head != "categorical":
            raise NotImplementedError(
                f"cell_head={cell_head!r} is a reserved ablation arm; "
                "see docs/ablation_hier_action_head.md"
            )
        cell_num = (int(config.shape[0]) - 2) // 4
        unimix = float(getattr(config.dist, "unimix_ratio", 0.0))
        self.cell_num = cell_num
        self.inner = CardQueryActor(config, inp_dim, default_card_table(), cell_num, unimix)

    def forward(self, x, mask=None, hand_ids=None):
        assert hand_ids is not None, "CardActionHead requires obs['hand_ids']"
        return CardFlatDist(self.inner(x, hand_ids, mask), self.cell_num)
