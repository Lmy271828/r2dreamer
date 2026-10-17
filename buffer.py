from collections import deque

import torch
from tensordict import TensorDict
from torchrl.data.replay_buffers import LazyTensorStorage, ReplayBuffer
from torchrl.data.replay_buffers.samplers import SliceSampler


class OnlineQueue:
    """Windows set aside at collection time so training can use them immediately.

    Holds the data itself rather than reading it back from storage, whose layout does not
    keep an env's consecutive steps adjacent.
    """

    def __init__(self, window, max_windows):
        self.window = int(window)
        self._pending = {}
        # Keep at most two training batches; drop the oldest windows if collection
        # outruns training so the queue cannot grow without bound.
        self.queue = deque(maxlen=int(max_windows))

    def push(self, data, coords):
        # env identity is the row position in `data`; coords only says where it landed
        where = {int(env): int(row) for row, env in coords.tolist()}
        for env in range(data.batch_size[0]):
            pending = self._pending.setdefault(env, [])
            pending.append((data[env].clone(), where[env]))
            if len(pending) == self.window:
                steps, rows = zip(*pending)
                self.queue.append((torch.stack(steps), env, torch.tensor(rows)))
                pending.clear()

    def take(self, count):
        """Stacked (data, env, rows) for `count` windows, or None while fewer are ready."""
        if len(self.queue) < count:
            return None
        windows = [self.queue.popleft() for _ in range(count)]
        data, envs, rows = zip(*windows)
        return torch.stack(data), torch.tensor(envs), torch.stack(rows)


class Buffer:
    def __init__(self, config):
        self.device = torch.device(config.device)
        self.storage_device = torch.device(config.storage_device)
        self.batch_size = int(config.batch_size)
        self.batch_length = int(config.batch_length)
        self.num_eps = 0
        self._buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=config.max_size, device=self.storage_device, ndim=2),
            sampler=SliceSampler(
                num_slices=self.batch_size, end_key=None, traj_key="episode", truncated_key=None, strict_length=True
            ),
            prefetch=0,
            batch_size=self.batch_size * (self.batch_length + 1),  # +1 for context
        )
        self._online = OnlineQueue(self.batch_length + 1, 2 * self.batch_size) if config.online else None
        self._update_latents = bool(config.update_latents)

    def add_transition(self, data):
        # This is batched data and lifted for storage.
        # (B, ...) -> (B, 1, ...)
        coords = self._buffer.extend(data.unsqueeze(1))
        if self._online is not None:
            self._online.push(data, coords)

    def sample(self, prefer_online=True):
        # Freshly collected windows take precedence for training. Logging can sample
        # regular replay without consuming a window reserved for learning.
        ready = self._online.take(self.batch_size) if prefer_online and self._online is not None else None
        if ready is not None:
            return self._online_batch(ready)
        sample_td, info = self._buffer.sample(return_info=True)
        # The sampler returns a flattened batch of length B*(T+1).
        # (B*(T+1), ...) -> (B, T+1, ...)
        sample_td = sample_td.view(-1, self.batch_length + 1)
        sample_td = self._to_device(sample_td)
        # The initial ones are used only to extract the latent vector
        initial = (sample_td["stoch"][:, 0], sample_td["deter"][:, 0])
        data = sample_td[:, 1:]
        data.set_("action", sample_td["action"][:, :-1].clone())  # action is 1 step back
        index = [ind.view(-1, self.batch_length + 1)[:, 1:] for ind in info["index"]]
        return data, index, initial

    def _to_device(self, sample_td):
        # A stacked TensorDict reports device None, so read it off an entry: the configured
        # storage device is not necessarily where the data actually sits.
        src_dev = sample_td.device or next(iter(sample_td.values())).device
        if src_dev == self.device:
            return sample_td
        if src_dev.type == "cpu" and self.device.type == "cuda":
            return sample_td.pin_memory().to(self.device, non_blocking=True)
        return sample_td.to(self.device, non_blocking=True)

    def _online_batch(self, ready):
        sample_td, envs, rows = ready
        sample_td = self._to_device(sample_td)
        initial = (sample_td["stoch"][:, 0], sample_td["deter"][:, 0])
        data = sample_td[:, 1:]
        data.set_("action", sample_td["action"][:, :-1].clone())
        envs = envs.to(self.device).unsqueeze(-1).expand(-1, self.batch_length)
        return data, [rows[:, 1:].to(self.device), envs], initial

    def update(self, index, stoch, deter):
        # Off means training keeps seeing the latents recorded at collection time.
        if not self._update_latents:
            return
        # Flatten the data
        index = [ind.reshape(-1) for ind in index]
        # (B, T, S, K) -> (B*T, S, K)
        stoch = stoch.reshape(-1, *stoch.shape[2:])
        # (B, T, D) -> (B*T, D)
        deter = deter.reshape(-1, *deter.shape[2:])
        # In storage, the length is the first dimension, and the batch (number of environments) is the second dimension.
        n = index[0].shape[0]
        self._buffer[index[1], index[0]] = TensorDict({"stoch": stoch, "deter": deter}, batch_size=(n,))

    def count(self):
        if self._buffer.storage.shape is None:
            return 0
        return self._buffer.storage.shape.numel()
