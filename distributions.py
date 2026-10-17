import torch
from torch import distributions as torchd
from torch.nn import functional as F

from tools import to_f32, to_i32


def symlog(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x):
    return torch.sign(x) * torch.expm1(torch.abs(x))


def unimix_logits(logits, ratio):
    """Mix `ratio` of a uniform into the categorical, returning log-probabilities."""
    # (..., K) -> (..., K)
    probs = F.softmax(to_f32(logits), dim=-1)
    probs = probs * (1.0 - ratio) + torch.ones_like(probs, dtype=torch.float32) * (ratio / probs.shape[-1])
    return torch.log(probs)


class OneHotDist(torchd.one_hot_categorical.OneHotCategorical):
    def __init__(self, logits, unimix_ratio=0.0):
        # (..., K)
        super().__init__(logits=unimix_logits(logits, unimix_ratio))

    @property
    def mode(self):
        # (..., K) Hard argmax forward, unperturbed-softmax jacobian backward, as in rsample.
        probs = torch.softmax(self.logits, -1)
        value = F.one_hot(torch.argmax(self.logits, dim=-1), self.logits.shape[-1]).to(probs.dtype)
        return value.detach() + (probs - probs.detach())

    def rsample(self, sample_shape=(), temperature=1.0):
        # (..., K) Hard categorical draw forward, unperturbed-softmax jacobian backward.
        # (probs - probs.detach()) is parenthesised deliberately: it is exactly 0, while
        # value + probs - probs.detach() would lose ~6e-8 in float32.
        probs = torch.softmax(self.logits, -1)
        idx = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1).reshape(probs.shape[:-1])
        value = F.one_hot(idx, probs.shape[-1]).to(probs.dtype)
        return value.detach() + (probs - probs.detach())

    def sample(self, **kwargs):
        raise NotImplementedError


class MultiOneHotDist:
    def __init__(self, logits, shape, unimix_ratio=0.0):
        self.shape = shape
        splits = torch.split(logits, shape, dim=-1)
        self.onehots = [OneHotDist(s, unimix_ratio=unimix_ratio) for s in splits]

    @property
    def mode(self):
        _modes = [dist.mode for dist in self.onehots]
        return torch.cat(_modes, dim=-1)

    def rsample(self, sample_shape=()):
        _rsamples = [dist.rsample() for dist in self.onehots]
        return torch.cat(_rsamples, dim=-1)

    def sample(self, **kwargs):
        raise NotImplementedError

    def log_prob(self, value):
        splits = torch.split(value, self.shape, dim=-1)
        _log_probs = [dist.log_prob(s) for dist, s in zip(self.onehots, splits)]
        return sum(_log_probs)

    def entropy(self):
        _entropies = [dist.entropy() for dist in self.onehots]
        return sum(_entropies)


class TwoHot:
    def __init__(self, logits, bins, squash=None, unsquash=None):
        # (..., N_bins), (N_bins,)
        self.logits = to_f32(logits)
        assert self.logits.shape[-1] == len(bins), (self.logits.shape, len(bins))

        self.bins = bins
        self.probs = F.softmax(self.logits, dim=-1)  # (..., N_bins)
        self.squash = squash if squash is not None else (lambda x: x)
        self.unsquash = unsquash if unsquash is not None else (lambda x: x)

    def mode(self):
        # (..., N_bins), (N_bins,) -> (..., 1)
        # The symexp bins are exactly antisymmetric (bins[n-1-i] == -bins[i]), so
        #     sum_i p_i * b_i  ==  sum_{i<n/2} (p_i - p_{n-1-i}) * b_i  (+ centre)
        # Cancelling on probabilities, not on ~1e8 products, survives compile reassociation.
        n = self.logits.shape[-1]
        m = n // 2
        p_lo = self.probs[..., :m]
        p_hi = self.probs[..., n - m :].flip(dims=(-1,))
        wavg = ((p_lo - p_hi) * self.bins[..., :m]).sum(dim=-1, keepdim=True)
        if n % 2 == 1:
            wavg = wavg + self.probs[..., m : m + 1] * self.bins[..., m : m + 1]
        return self.unsquash(wavg)

    def log_prob(self, target):
        # (..., 1)
        assert target.dtype == self.probs.dtype
        target = target.squeeze(-1)  # (...,)
        target_squashed = self.squash(target).detach()  # (...,)
        # below/above: (...,)
        below = to_i32(self.bins <= target_squashed.unsqueeze(-1)).sum(dim=-1) - 1
        above = len(self.bins) - to_i32(self.bins > target_squashed.unsqueeze(-1)).sum(dim=-1)
        below = torch.clamp(below, 0, len(self.bins) - 1)
        above = torch.clamp(above, 0, len(self.bins) - 1)
        equal = below == above
        dist_to_below = torch.where(
            equal,
            torch.tensor(1.0, device=target.device, dtype=torch.float32),
            (self.bins[below] - target_squashed).abs(),
        )
        dist_to_above = torch.where(
            equal,
            torch.tensor(1.0, device=target.device, dtype=torch.float32),
            (self.bins[above] - target_squashed).abs(),
        )
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        oh_below = to_f32(F.one_hot(below, num_classes=len(self.bins)))
        oh_above = to_f32(F.one_hot(above, num_classes=len(self.bins)))
        # (..., N_bins)
        mixed_target = oh_below * weight_below.unsqueeze(-1) + oh_above * weight_above.unsqueeze(-1)
        log_pred = self.logits - torch.logsumexp(self.logits, dim=-1, keepdim=True)  # (..., N_bins)
        return (mixed_target * log_pred).sum(dim=-1)  # (...)


class MSEDist:
    def __init__(self, mode, agg="sum"):
        # (..., D)
        self._mode = to_f32(mode)
        self._agg = agg

    def mode(self):
        return self._mode

    def mean(self):
        return self._mode

    def log_prob(self, value):
        # (..., D)
        assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
        assert self._mode.dtype == value.dtype, (self._mode.dtype, value.dtype)
        distance = (self._mode - value) ** 2
        if self._agg == "mean":
            loss = distance.mean(list(range(len(distance.shape)))[2:])
        elif self._agg == "sum":
            loss = distance.sum(list(range(len(distance.shape)))[2:])
        else:
            raise NotImplementedError(self._agg)
        return -loss  # (...)


class SymlogDist:
    def __init__(self, mode, dist="mse", agg="sum", tol=1e-8):
        # (..., D)
        self._mode = to_f32(mode)
        self._dist = dist
        self._agg = agg
        self._tol = tol

    def mode(self):
        return symexp(self._mode)

    def mean(self):
        return symexp(self._mode)

    def log_prob(self, value):
        # (..., D)
        assert self._mode.shape == value.shape
        assert self._mode.dtype == value.dtype
        if self._dist == "mse":
            distance = (self._mode - symlog(value)) ** 2.0
            distance = torch.where(distance < self._tol, 0, distance)
        elif self._dist == "abs":
            distance = torch.abs(self._mode - symlog(value))
            distance = torch.where(distance < self._tol, 0, distance)
        else:
            raise NotImplementedError(self._dist)
        if self._agg == "mean":
            loss = distance.mean(list(range(len(distance.shape)))[2:])
        elif self._agg == "sum":
            loss = distance.sum(list(range(len(distance.shape)))[2:])
        else:
            raise NotImplementedError(self._agg)
        return -loss  # (...)


class Bound:
    def __init__(self, dist):
        super().__init__()
        self._dist = dist

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def entropy(self):
        return self._dist.entropy()

    @property
    def mode(self):
        out = self._dist.mean
        return out / torch.clip(torch.abs(out), min=1.0).detach()

    def sample(self, sample_shape=()):
        out = self._dist.rsample(sample_shape)
        return out / torch.clip(torch.abs(out), min=1.0).detach()

    def log_prob(self, x):
        return self._dist.log_prob(x)


def bounded_normal(x, min_std, max_std, **kwargs):
    mean, std = torch.chunk(x, 2, dim=-1)
    std = (max_std - min_std) * torch.sigmoid(std + 2.0) + min_std
    # NOTE: Bound can be added
    dist = torchd.normal.Normal(torch.tanh(to_f32(mean)), to_f32(std))
    return torchd.independent.Independent(dist, 1)


def normal_std_fixed(mean, std, **kwargs):
    dist = torchd.normal.Normal(to_f32(mean), to_f32(std))
    return Bound(torchd.independent.Independent(dist, 1))


def onehot(mean, unimix_ratio, **kwargs):
    return OneHotDist(to_f32(mean), unimix_ratio=unimix_ratio)


def multi_onehot(mean, unimix_ratio, shape, **kwargs):
    return MultiOneHotDist(to_f32(mean), shape, unimix_ratio=unimix_ratio)


def binary(logits, **kwargs):
    return torchd.independent.Independent(torchd.bernoulli.Bernoulli(logits=to_f32(logits)), 1)


def symexp_twohot(logits, bin_num, **kwargs):
    if bin_num % 2 == 1:
        half = torch.linspace(-20, 0, (bin_num - 1) // 2 + 1, dtype=torch.float32, device=logits.device)
        half = symexp(half)
        bins = torch.concatenate([half, -half[:-1].flip(dims=(0,))], 0)
    else:
        half = torch.linspace(-20, 0, bin_num // 2, dtype=torch.float32, device=logits.device)
        half = symexp(half)
        bins = torch.concatenate([half, -half.flip(dims=(0,))], 0)
    return TwoHot(to_f32(logits), bins)


def symlog_mse(logits, **kwargs):
    return SymlogDist(to_f32(logits))


def mse(logits, **kwargs):
    return MSEDist(to_f32(logits))


def identity(logits, **kwargs):
    return logits


def kl(logits_left, logits_right):
    # (..., K), (..., K)
    logprob_left = torch.log_softmax(logits_left, -1)
    logprob_right = torch.log_softmax(logits_right, -1)
    prob = torch.softmax(logits_left, -1)
    return (prob * (logprob_left - logprob_right)).sum(-1)  # (...)
