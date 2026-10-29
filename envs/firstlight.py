"""Clash Royale via the FirstLight offline engine, image observations.

One environment step = one 5-tick decision (250 ms of game time at speed 1.0;
run the engine at `speed < 1.0` to leave capture headroom). The battle is
rendered to the MuMu guest surface (rendering must stay ON -- `render off`
freezes eglSwapBuffers and screenshots would go stale), and every step ends
with an `adb screencap` of the calibrated arena rect, downsampled to `size`.

Action: Discrete(1 + 4*576 + 1). Index 0 is WAIT; index `1 + slot*576 + cell`
(cell = y*18 + x) plays the card currently in `slot` at that grid cell.
Index 2305 (ABILITY_INDEX) activates the deployed ability of the card:
legal only when the engine reports a ready type-2 activation
(action_mask.ability_sources), and degrades to WAIT if the entity snapshot
is stale. Illegal indices degrade to WAIT; dynamic masking is layered onto
the actor in the M2 changes (the flattened mask is exposed as
obs["action_mask"]).

Structured side channels (encoded by the MLP encoder, not the CNN):
  elixir      -- own elixir, straight from the probe (authoritative:
                 covers Elixir Collector, Elixir Golem refunds and every
                 other event-driven gain). This is a probe-dependent
                 channel; a deployment-faithful rule ledger was tried and
                 removed -- it cannot see event-driven elixir (see git
                 history, ElixirLedger).
  elixir_rate -- current elixir growth rate in elixir/second (rule-based:
                 the probe exposes no rate; collector ticks are absent here)
  hand_ids    -- current hand card pool indices (4, 0 = empty/UNK)
  deck_ids    -- own deck pool indices (8) for critic deck conditioning

Continuous collection mode: the engine runs in real time and is never
paused between steps; each step commits the action for the next 5 ticks
and returns the post-action observation. Frames come from `StreamCapture`
(a background screenrecord h264 stream, ~5 ms fresh frames) when
`capture_backend="stream"`, with the synchronous adb screencap kept as
fallback whenever the stream is stale or unavailable.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

FIRSTLIGHT_ROOT = os.environ.get(
    "CR_FIRSTLIGHT_ROOT", "D:/pynoob/FirstLight/FirstLight_CR"
)
if FIRSTLIGHT_ROOT not in sys.path:
    sys.path.insert(0, FIRSTLIGHT_ROOT)

from native_runner.battle_env import BattleEnvError  # noqa: E402 (path injected above)
from native_runner.cr_native_env import RunnerError  # noqa: E402 (path injected above)

DECKS = {
    # 群卡研究组态：一次部署产生多个实体，因果组结构清晰
    "swarm": (26000012, 26000041, 26000022, 26000010, 26000007, 28000010, 28000000, 26000000),
    "standard": (26000000, 26000001, 26000005, 28000001, 28000000, 26000003, 26000014, 26000018),
}
def _load_card_names():
    try:
        from native_runner.card_names import CARD_NAMES

        return set(CARD_NAMES)
    except Exception:
        return set()


CARD_NAMES_ALL = _load_card_names()
FIRST_PLAYABLE_TICK = 90
GRID_W, GRID_H = 18, 32
CELLS = GRID_W * GRID_H
WAIT_INDEX = 0
HAND_NUM = 4
# 技能位（预留接口的扁平编码）：index 2305 = ACTIVATE_ABILITY。
# 合法性由掩码末位给出（引擎 action_mask.ability_sources 非空），step 时
# 解析出 (source_entity, ability_id) 提交 type-2 激活（无目标载荷）。
ABILITY_INDEX = 1 + HAND_NUM * CELLS
ACTION_DIM = ABILITY_INDEX + 1
ARENA_CROP = (111, 430, 967, 1620)


def _load_deck_pool():
    """对手侧随机卡组池 = 卡牌目录 ∩ 竞技支持白名单（剔除 Super*/禁用卡）。

    账号卡池已解锁（T4 实测双侧全随机通过）：池 = 全部受支持卡牌，
    deck0/deck1 都可用。
    """
    ids = set(CARD_NAMES_ALL)
    try:
        from native_runner.card_names import CARD_NAMES

        ids &= set(CARD_NAMES)
    except Exception:
        pass
    try:
        sup = json.loads(
            (Path(FIRSTLIGHT_ROOT) / "native_runner" / "data" / "competitive" / "card_support.json").read_text()
        )
        supported = {int(c["card_id"]) for c in sup.get("supported_cards", [])}
        ids &= supported
        # 注：27000 段是建筑卡（加农炮/特斯拉等），26000 段内混有冠军卡
        # （Monk 26000077、Little Prince 26000093）。冠军技能动作
        # （ACTIVATE_ABILITY）不在当前 2305 路动作空间里，部署本身合法。
    except Exception:
        pass
    return tuple(sorted(ids)) if ids else ()


DECK_POOL = _load_deck_pool()
DECK_INDEX = {card: i for i, card in enumerate(DECK_POOL)}                 # 对手侧全支持池（122）
DECK_POOL_OWNED = tuple(sorted(set(DECKS["swarm"]) | set(DECKS["standard"])))


class FirstLightInlineEnv:
    """In-process N-env adapter mirroring ParallelEnv's trainer surface.

    Bypasses the multiprocessing pipe (it deadlocked in long sessions).
    With multiple local engines (one MuMu guest per env), a step submits
    all envs through a thread pool: each env's wall time is dominated by
    socket reads and frame decoding that release the GIL, so the waits
    genuinely overlap and aggregate throughput scales near-linearly.

    Mirrors ParallelEnv semantics the trainer relies on:
      step(action, done) -> (TensorDict batch_size=(N,), done (N,))
      lazy auto-reset: on per-env done, the next step resets that env and
      returns reward 0 / is_first True for its row
      scalar obs lifted to (N, 1) via lift-dim
    """

    def __init__(self, make_env_fn, device, env_num=1):
        import concurrent.futures

        import torch
        from tensordict import TensorDict

        self._torch = torch
        self._TensorDict = TensorDict
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=env_num)
        self._envs = [make_env_fn(i) for i in range(env_num)]
        self.observation_space = self._envs[0].observation_space
        self.action_space = self._envs[0].action_space
        self.env_num = env_num
        self._needs_reset = [True] * env_num
        self._last_obs1 = [None] * env_num

    # -- per-env worker (runs in the pool thread) -------------------------

    def _step_one(self, index, onehot, onehot1, done_value):
        env = self._envs[index]
        if done_value or self._needs_reset[index]:
            obs = env.reset()
            self._needs_reset[index] = False
            trans = dict(obs)
            trans["reward"] = np.float32(0.0)
            self._last_obs1[index] = getattr(env, "_last_obs1", None)
            return trans, False
        obs, reward, done_flag, info = env.step(onehot, action1=onehot1)
        # 终局/截断/基础设施中止后，下一步必须重置该 env
        self._needs_reset[index] = bool(done_flag)
        trans = dict(obs)
        trans["reward"] = np.float32(reward)
        self._last_obs1[index] = getattr(env, "_last_obs1", None)
        return trans, bool(done_flag)

    def step(self, action, done, action1=None):
        torch = self._torch
        act = action.detach().reshape(self.env_num, -1).cpu()
        act1 = None
        if action1 is not None:
            act1 = action1.detach().reshape(self.env_num, -1).cpu()
        if hasattr(done, "flatten"):
            done_flags = [bool(v) for v in done.flatten().cpu().tolist()]
        else:
            done_flags = [bool(v) for v in np.atleast_1d(done)]

        futures = []
        for i in range(self.env_num):
            index = int(torch.argmax(act[i]).item())
            onehot = np.zeros(self.action_space.shape[0], dtype=np.float32)
            onehot[index] = 1.0
            onehot1 = None
            if act1 is not None:
                index1 = int(torch.argmax(act1[i]).item())
                onehot1 = np.zeros(self.action_space.shape[0], dtype=np.float32)
                onehot1[index1] = 1.0
            futures.append(self._pool.submit(self._step_one, i, onehot, onehot1, done_flags[i]))
        results = [f.result() for f in futures]

        keys = results[0][0].keys()
        stacked = {
            k: torch.as_tensor(np.stack([r[0][k] for r in results]), device="cpu")
            for k in keys
        }
        td = self._TensorDict(stacked, batch_size=(self.env_num,), device="cpu")
        for key in td.keys(include_nested=True):
            if td[key].ndim == 1:
                td[key] = td[key].unsqueeze(-1)
        step_done = torch.tensor([r[1] for r in results], dtype=torch.bool)
        return td, step_done

    def reset(self):
        self._needs_reset = [True] * self.env_num
        self._last_obs1 = [None] * self.env_num

    def pop_obs1(self):
        """M5：取走本步的 owner-1 视图观测（TensorDict (B,1,*)）；缺任一即 None。"""
        torch = self._torch
        if not hasattr(self, "_last_obs1") or any(o is None for o in self._last_obs1):
            return None
        keys = self._last_obs1[0].keys()
        stacked = {
            k: torch.as_tensor(np.stack([o[k] for o in self._last_obs1]), device="cpu")
            for k in keys
        }
        td = self._TensorDict(stacked, batch_size=(self.env_num,), device="cpu")
        for key in td.keys(include_nested=True):
            if td[key].ndim == 1:
                td[key] = td[key].unsqueeze(-1)
        self._last_obs1 = [None] * self.env_num  # 取走即清
        return td


class FirstLight(gym.Env):
    """Image-observation wrapper around the FirstLight offline engine."""

    metadata = {}

    def __init__(
        self,
        task="swarm",
        size=(176, 128),
        seed=0,
        speed=0.5,
        capture_backend="adb",
        serial=None,
        control_port=None,
        deck_mode="fixed",
        opp_policy="afk",
        vm_index=None,
        vm_name=None,
    ):
        self._task = task
        self._deck0 = DECKS.get(task, DECKS["swarm"])
        self._deck1 = None  # reset 时由 _ensure_imports 填 standard
        self._size = (int(size[0]), int(size[1]))  # (H, W)
        self._seed_base = int(seed)
        self._speed = float(speed)
        self._capture_backend = str(capture_backend)
        self._deck_mode = str(deck_mode)
        self._opp_policy = str(opp_policy)
        self._opp_p_wait = 0.3
        self._opp_plays = 0
        self._opp_last = "init"
        self._swap_next = False  # 换边双打：下局 deck0/deck1 互换
        self._deck_override = None   # 换边双打 harness：显式指定下局双方卡组
        self._seed_override = None
        # 多实例：serial/control_port 覆盖 .env 的单实例默认值
        self._serial_override = serial
        self._vm_index = int(vm_index) if vm_index is not None else None
        self._vm_name = vm_name
        self._control_port = int(control_port) if control_port else None
        self._binding_fingerprint = None
        # M1 对齐旁路日志：每步记录 (tick, 帧龄, 流状态)，量化图像↔tick 组内偏移
        self._last_frame_meta = None
        self._align_path = None
        self._stream = None
        self._capture_fallbacks = 0
        self._last_obs = None
        self._last_elixir = 5.0  # 圣水通道缺读数时的回退值（开局标准值）
        self._ability_by_entity = {}  # canonical 实体 -> 就绪技能 id（上一步 obs 快照）
        self._ability_by_entity_1 = {}  # owner-1 同上（自博弈对手技能解析）
        self._mask1 = None  # owner-1 掩码（自博弈）
        self._last_obs1 = None  # owner-1 视图观测（trainer 经 pop_obs1 取用）
        self._episodes = 0
        self._battle = None
        self._mask = None
        self.reward_range = (-np.inf, np.inf)

    # -- engine plumbing --------------------------------------------------

    def _ensure_imports(self):
        from native_runner.battle_env import BattleEnvError, BattleEnvV1
        from native_runner.contracts import ActionV1
        from native_runner.local_config import load_env, setting
        from native_runner.match_factory import MatchConfig, STANDARD_DECK

        load_env()
        self._battle_type = BattleEnvV1
        self._action_v1 = ActionV1
        self._match_config = MatchConfig
        self._standard_deck = tuple(STANDARD_DECK)
        self._adb = setting("CR_ADB") or "adb"
        self._serial = self._serial_override or setting("CR_ADB_SERIAL") or "127.0.0.1:16416"
        self._control_port = self._control_port or int(setting("CR_CONTROL_PORT", "26789"))

    def _adb_cmd(self, *args, timeout=10.0):
        return subprocess.run(
            [self._adb, *args], capture_output=True, text=True, timeout=timeout, check=False
        )

    def _verify_binding(self):
        """实例绑定校验：serial 在线 + 控制端口的 forward 归属正确。

        防"连错实例"：控制端口被残留实例的 forward 占用、或 serial 被换成
        另一个 MuMu guest 时，探针照样能连上，但轨迹来自错误引擎。校验失败
        抛 RuntimeError，由调用方走 start_offline 重启（它会重建正确的
        forward 并校验 VM 名）。
        """
        from native_runner.local_config import setting
        listed = self._adb_cmd("devices").stdout.splitlines()
        rows = [line.split() for line in listed if len(line.split()) == 2]
        state = next((s for serial, s in rows if serial == self._serial), None)
        if state != "device":
            raise RuntimeError(
                f"adb serial {self._serial} not online (state={state!r}); "
                f"visible: {[r[0] for r in rows]}"
            )
        guest_port = int(setting("CR_GUEST_CONTROL_PORT", "26789"))
        fwd = [
            line.split()
            for line in self._adb_cmd("-s", self._serial, "forward", "--list").stdout.splitlines()
            if len(line.split()) >= 3
        ]
        host, guest = f"tcp:{self._control_port}", f"tcp:{guest_port}"
        if not any(r[0] == self._serial and r[1] == host and r[2] == guest for r in fwd):
            raise RuntimeError(
                f"control port {self._control_port} forward missing/mismatched for "
                f"{self._serial} (expected {host}->{guest}; got {fwd})"
            )
        if getattr(self, "_binding_fingerprint", None) != self._serial:
            model = (self._adb_cmd("-s", self._serial, "shell", "getprop", "ro.product.model").stdout or "").strip()
            self._binding_fingerprint = self._serial
            print(f"FirstLight: bound to {self._serial} ({model}), control {host}->{guest}", flush=True)

    def _ensure_engine(self, restart: bool = False):
        import socket

        if not restart:
            with socket.socket() as sock:
                sock.settimeout(2.0)
                if sock.connect_ex(("127.0.0.1", self._control_port)) == 0:
                    try:
                        self._verify_binding()
                        return
                    except (RuntimeError, subprocess.TimeoutExpired) as error:
                        print(
                            f"FirstLight: binding check failed ({str(error)[:160]}); "
                            "re-provisioning engine...",
                            flush=True,
                        )
                        # 落到下方 start_offline：重启并重建 forward
        script = Path(FIRSTLIGHT_ROOT) / "native_runner" / "start_offline.ps1"
        cmd = [
            "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
            "-File", str(script),
            "-ControlPort", str(self._control_port),
            "-Serial", self._serial,
        ]
        if self._vm_index is not None:
            cmd += ["-VmIndex", str(self._vm_index)]
        if self._vm_name:
            cmd += ["-VmName", self._vm_name]
        subprocess.run(cmd, check=False, timeout=900)

    def _ensure_battle(self, restart: bool = False):
        if self._battle is None:
            self._ensure_imports()
            self._ensure_engine(restart=restart)
            if self._control_port:
                # 多实例：每个 env 的 probe 经 adb forward 映射到不同宿主端口
                from native_runner.cr_native_env import NativeClashEnv

                native = NativeClashEnv(host="127.0.0.1", port=int(self._control_port))
                self._battle = self._battle_type(native=native)
            else:
                self._battle = self._battle_type()

    # -- capture ----------------------------------------------------------

    def _capture(self):
        from native_runner.vision.capture import capture_arena

        if self._stream is not None:
            image, meta = self._stream.latest()
            # 屏内 5 tick（speed 0.5 = 500ms）内的新鲜帧直接采用；流重启窗口
            # （screenrecord 180s 硬上限，~2s 间隙）或冷启动期退回 adb 截图。
            if image is not None and meta["age_s"] is not None and meta["age_s"] < 0.6:
                self._last_frame_meta = {"age_s": meta["age_s"], "frames": meta["frames"], "stream": True}
                return image
            self._capture_fallbacks += 1
            if self._capture_fallbacks % 50 == 1:
                print(
                    f"FirstLight: stream frame stale/unavailable "
                    f"(age={meta['age_s']}s, fallbacks={self._capture_fallbacks}, "
                    f"stats={meta}); using adb screencap",
                    flush=True,
                )

        self._last_frame_meta = {"stream": False}
        last_error = None
        for attempt in range(3):
            try:
                return capture_arena(
                    self._adb, self._serial, self._size, timeout=15.0
                )
            except Exception as error:  # noqa: BLE001 -- adb 偶发挂起/重置，重试一次
                last_error = error
        raise last_error

    def _scripted_action(self, native_now=None):
        """随机合法脚本对手：p_wait 概率等待，否则从 actor 1 的合法 (槽,格) 均匀抽一个。

        native_now 由调用方传入（与 player-0 的手牌查询共用一次 observe），
        避免在提交关键路径上多付一次探针往返——那正是 not-queued 竞态的来源。
        """
        try:
            rng = np.random.default_rng()
            mask1 = self._battle.action_mask(1)
            # 决策窗未就绪的掩码只有 kinds/reasons、没有 hand_slots——此时
            # 若提交 wait 会把 owner-1 死线再推 10 tick，造成永久 AFK 锁。
            # 返回 None = 本窗不提交，死线自然过期，下一窗自愈就绪。
            if getattr(mask1, "hand_slots", None) is None:
                self._opp_last = "not_ready_skip"
                return None
            if rng.random() < self._opp_p_wait:
                self._opp_last = "rng_wait"
                return self._action_v1.wait(1, ticks=10, execute_offset_ticks=3)
            choices = []
            for slot in range(4):
                if not mask1.hand_slots[slot] or str(slot) not in mask1.placement_masks:
                    continue
                rows = mask1.placement_masks[str(slot)]["row_major"]
                for cy, row in enumerate(rows):
                    for cx, legal in enumerate(row):
                        if legal:
                            choices.append((slot, cx, cy))
            if not choices:
                self._opp_last = "empty"
                return self._action_v1.wait(1, ticks=10, execute_offset_ticks=3)
            slot, cx, cy = choices[int(rng.integers(len(choices)))]
            try:
                hand = next(p for p in (native_now or self._battle.native.observe())["players"] if p["owner"] == 1)["hand"]
                card_id = next(c["cardId"] for c in hand if c["handIndex"] == slot)
            except Exception:  # noqa: BLE001 -- 手牌读取失败时引擎通常也接受缺省
                card_id = 0
            self._opp_plays += 1
            self._opp_last = "play"
            return self._action_v1.play(
                owner=1, hand_slot=slot, grid=(cx, cy), card_id=card_id, next_decision_ticks=10,
                execute_offset_ticks=3,
            )
        except Exception as error:  # noqa: BLE001 -- 对手故障绝不带崩采集
            self._opp_last = f"err:{str(error)[:60]}"
            return self._action_v1.wait(1, ticks=10, execute_offset_ticks=3)

    def _align_log_tick(self, tick, engine_elixir=None):
        """M1 对齐旁路：被动记录每步的引擎 tick 与帧龄。

        图像内容 tick ≈ obs0.tick - age_s × speed / 0.05（帧在流管线里晚到
        age_s 秒）；本日志就是该偏移的原始测量，供 M1 离线分析分布。
        """
        if self._align_path is None:
            port = int(self._control_port or 26789)
            default = str(Path(FIRSTLIGHT_ROOT).parent / "align_log" / f"env_{port}.jsonl")
            self._align_path = Path(os.environ.get("CR_ALIGN_LOG", default))
            self._align_path.parent.mkdir(parents=True, exist_ok=True)
        meta = self._last_frame_meta or {}
        line = json.dumps({
            "t": round(time.time(), 4),
            "tick": int(tick),
            "age_s": meta.get("age_s"),
            "stream": bool(meta.get("stream", False)),
            "speed": self._speed,
            "opp_plays": self._opp_plays,
            "opp_last": self._opp_last,
            "elixir": round(float(engine_elixir or 0.0), 3),
        })
        with open(self._align_path, "a") as f:
            f.write(line + "\n")

    def _ensure_stream(self):
        if self._capture_backend != "stream" or self._stream is not None:
            return
        from native_runner.vision.capture import StreamCapture

        self._stream = StreamCapture(self._adb, self._serial, self._size)
        self._stream.start()

    # -- helpers ----------------------------------------------------------

    def _sync_step(self, action):
        """One engine decision with the human-latency sync shim.

        resume -> throwaway WAIT resyncs the env's internal clock to the
        engine tick (the match would otherwise drift ahead of cached state)
        -> submit the real action -> freeze.
        """

        battle = self._battle
        battle.native.resume()
        battle.step_actor(0, {0: [self._action_v1.wait(0)]})
        if action is None:
            result = battle.step_actor(0, {0: [self._action_v1.wait(0)]})
            battle.native.pause()
            return result

        from dataclasses import replace

        last_error = None
        for attempt in range(3):
            offset = max(1, int(action.execute_offset_ticks or 0)) + attempt * 2
            try:
                result = battle.step_actor(
                    0, {0: [replace(action, execute_offset_ticks=offset)]}
                )
                battle.native.pause()
                return result
            except Exception as error:  # noqa: BLE001 -- probe races surface as RunnerError
                last_error = error
                if "not queued" not in str(error) or attempt == 2:
                    raise
                battle.step_actor(0, {0: [self._action_v1.wait(0)]})
        raise last_error

    def _elixir_rate(self, tick):
        from native_runner.battle_env import _elixir_raw_per_tick

        return float(_elixir_raw_per_tick(int(tick))) * 20.0 / 10000.0

    def _probe_elixir(self, player, tick):
        """圣水通道：探针真值（覆盖收集器涌出等一切事件式产出）。

        探针缺读数（理论不发生）时退回上一次读数，保证 obs 键恒有值。
        """
        exact = getattr(player, "elixir_exact", None)
        if exact is not None:
            self._last_elixir = float(np.clip(float(exact), 0.0, 10.0))
        return self._last_elixir

    def _mask_flat(self, mask=None):
        """扁平动作掩码；mask 缺省用 owner-0 的，自博弈对手传 owner-1 的。"""
        mask = self._mask if mask is None else mask
        flat = np.zeros(ACTION_DIM, dtype=np.uint8)
        flat[0] = 1  # WAIT is always legal
        if mask is not None:
            for slot in range(HAND_NUM):
                if not mask.hand_slots[slot]:
                    continue
                key = str(slot)
                if key not in mask.placement_masks:
                    continue  # e.g. card outside the semantic subset
                rows = mask.placement_masks[key]["row_major"]
                base = 1 + slot * CELLS
                for y, row in enumerate(rows):
                    for x, legal in enumerate(row):
                        if legal:
                            flat[base + y * GRID_W + x] = 1
            # 技能位：引擎判定存在合法的 type-2 激活候选（READY + 冷却 0 +
            # 圣水足够），step 时解析具体 (source_entity, ability_id)
            flat[ABILITY_INDEX] = 1 if len(tuple(mask.ability_sources)) > 0 else 0
        return flat

    def _decode(self, index):
        index = int(index)
        if index == ABILITY_INDEX:
            return -1, None, None  # 技能位哨兵：槽位 -1
        if index <= WAIT_INDEX or index > HAND_NUM * CELLS:
            return None, None, None
        offset = index - 1
        slot = offset // CELLS
        cell = offset % CELLS
        return slot, cell % GRID_W, cell // GRID_W

    def _legal(self, slot, cx, cy, mask=None):
        mask = self._mask if mask is None else mask
        if mask is None or slot is None:
            return False
        if slot == -1:  # 技能位
            return len(tuple(mask.ability_sources)) > 0
        if not mask.hand_slots[slot] or str(slot) not in mask.placement_masks:
            return False
        return bool(mask.placement_masks[str(slot)]["row_major"][cy][cx])

    def _selfplay_action(self, action1, native_now):
        """owner-1 动作：trainer 传来的扁平索引 → 引擎动作；None = 脚本热身。

        自博弈对手机制层使用探针特权（手牌/掩码读数）——对手是环境的
        一部分，见 purity_ladder.md P3；学习方的输入不含任何特权。
        """
        if action1 is None:
            return self._scripted_action(native_now)  # 热身期：随机脚本
        mask1 = self._battle.action_mask(1)
        if getattr(mask1, "hand_slots", None) is None:
            self._opp_last = "not_ready_skip"
            return None  # 决策窗未就绪：不提交，死线自然过期
        slot, cx, cy = self._decode(action1)
        if not self._legal(slot, cx, cy, mask1):
            self._opp_last = "selfplay_illegal"
            return self._action_v1.wait(1, ticks=10, execute_offset_ticks=3)
        if slot == -1:
            entity, ability_id = self._resolve_ability_owner(1, mask1)
            if entity is None:
                return self._action_v1.wait(1, ticks=10, execute_offset_ticks=3)
            self._opp_last = f"ability:{ability_id[:24]}"
            return self._action_v1(
                owner=1, kind="activate_ability", source_entity=entity,
                ability_id=ability_id, next_decision_ticks=10,
            )
        rp1 = next(p for p in native_now["players"] if p["owner"] == 1)
        hand_card = next((c for c in rp1["hand"] if c["handIndex"] == slot), None)
        if hand_card is None:
            return self._action_v1.wait(1, ticks=10, execute_offset_ticks=3)
        self._opp_plays += 1
        self._opp_last = f"selfplay:{slot}"
        return self._action_v1.play(
            owner=1, hand_slot=slot, grid=(cx, cy), card_id=hand_card["cardId"],
            next_decision_ticks=10,
        )

    def _resolve_ability_owner(self, owner, mask):
        table = self._ability_by_entity if owner == 0 else self._ability_by_entity_1
        for entity in sorted(set(int(s) for s in mask.ability_sources)):
            ability_id = table.get(entity)
            if ability_id is not None:
                return entity, ability_id
        return None, None

    def _resolve_ability(self):
        """把掩码的 ability_sources（canonical 实体）解析成 (entity, ability_id)。

        依据上一步 obs 合约里的 ability_runtime_states（canonical id 同源）。
        多个就绪实体取 canonical id 最小者（确定性；更聪明的选择是策略的
        事——扁平接口下一次只能释放一个）。解析失败返回 None（降级 WAIT）。
        """
        source = sorted(set(int(s) for s in self._mask.ability_sources))
        for entity in source:
            ability_id = self._ability_by_entity.get(entity)
            if ability_id is not None:
                return entity, ability_id
        return None

    def _track_abilities(self, obs0):
        """从 obs 合约刷新实体→就绪技能快照（_resolve_ability 的依据）。"""
        self._track_abilities_owner(obs0, 0)

    def _track_abilities_owner(self, obs0, owner):
        try:
            player = next(p for p in obs0.players if p.owner == owner)
            table = self._ability_by_entity if owner == 0 else self._ability_by_entity_1
            table.clear()
            table.update({
                int(st.source_entity): str(st.ability_id)
                for st in getattr(player, "ability_runtime_states", ())
                if st.source_entity is not None
            })
        except Exception:  # noqa: BLE001 -- 快照失败只影响技能位解析，不影响主流程
            (self._ability_by_entity if owner == 0 else self._ability_by_entity_1).clear()

    def _assemble(self, obs0, is_first, done, terminated):
        player = next(p for p in obs0.players if p.owner == 0)
        elixir = player.elixir_exact if player.elixir_exact is not None else 0.0
        return {
            "image": self._capture(),
            "is_first": np.bool_(is_first),
            "is_last": np.bool_(done),
            "is_terminal": np.bool_(terminated),
            "elixir": np.array([elixir], dtype=np.float32),
            "elixir_rate": np.array([self._elixir_rate(obs0.tick)], dtype=np.float32),
            "hand_ids": self._hand_ids(native_now),
            "action_mask": self._mask_flat(),
        }

    # -- gym API ----------------------------------------------------------

    @property
    def observation_space(self):
        height, width = self._size
        return gym.spaces.Dict(
            {
                "image": gym.spaces.Box(0, 255, (height, width, 3), dtype=np.uint8),
                "elixir": gym.spaces.Box(0, 10, (1,), dtype=np.float32),
                "elixir_rate": gym.spaces.Box(0, 1, (1,), dtype=np.float32),
                "hand_ids": gym.spaces.Box(0, 300, (4,), dtype=np.int64),
                "deck_ids": gym.spaces.Box(0, 300, (8,), dtype=np.int64),
                "action_mask": gym.spaces.Box(0, 1, (ACTION_DIM,), dtype=np.uint8),
            }
        )

    @property
    def action_space(self):
        return gym.spaces.Discrete(ACTION_DIM)

    def reset(self):
        """开局；基础设施故障时强制重启引擎并重试一次（捕获死亡也算）。"""
        try:
            return self._reset_inner()
        except (RunnerError, BattleEnvError, TimeoutError) as error:
            print(
                f"FirstLight: reset failed ({str(error)[:120]}); restarting offline engine...",
                flush=True,
            )
            self._battle = None
            return self._reset_inner(restart=True)

    def _hand_ids(self, native_now):
        """owner-0 当前手牌的卡池索引（空/未知 = 0 号 UNK），供 actor 卡头查嵌入。"""
        hand = next(p for p in native_now["players"] if p["owner"] == 0)["hand"]
        ids = [DECK_INDEX.get(c["cardId"], 0) for c in hand]
        return np.array((ids + [0] * 4)[:4], dtype=np.int64)

    def _hand_ids_owner(self, native_now, owner):
        hand = next(p for p in native_now["players"] if p["owner"] == owner)["hand"]
        ids = [DECK_INDEX.get(c["cardId"], 0) for c in hand]
        return np.array((ids + [0] * 4)[:4], dtype=np.int64)

    def _sample_decks(self, seed):
        """按 deck_mode 采样双方卡组（fixed 模式保持构造时的组态）。"""
        if self._deck_mode == "random":
            if not DECK_POOL:
                raise RuntimeError("deck_mode=random but card pool is empty")
            rng = np.random.default_rng(seed)
            self._deck0 = tuple(int(c) for c in rng.choice(DECK_POOL, size=8, replace=False))
            self._deck1 = tuple(int(c) for c in rng.choice(DECK_POOL, size=8, replace=False))
        elif self._deck_mode.startswith("random_opp"):
            pool = DECK_POOL_OWNED if self._deck_mode == "random_opp_owned" else DECK_POOL
            rng = np.random.default_rng(seed)
            self._deck1 = tuple(int(c) for c in rng.choice(pool, size=8, replace=False))

    def set_next_match(self, deck0, deck1, seed=None):
        """换边双打 harness：显式指定下一局的双方卡组与 seed（优先于采样）。"""
        self._deck_override = (tuple(int(c) for c in deck0), tuple(int(c) for c in deck1))
        self._seed_override = int(seed) if seed is not None else None

    def _reset_inner(self, restart=False):
        self._ensure_battle(restart=restart)
        self._ensure_stream()  # 流编码器预热（~1s）与引擎 reset/热身并行
        self._episodes += 1
        seed = self._seed_base + self._episodes * 7919
        if self._deck_override:
            self._deck0, self._deck1 = self._deck_override
            self._deck_override = None
            if self._seed_override is not None:
                seed = self._seed_override
                self._seed_override = None
        else:
            self._sample_decks(seed)
        deck0, deck1 = (self._deck1, self._deck0) if self._swap_next else (self._deck0, self._deck1)
        self._swap_next = False
        reset_kwargs = {
            "match_config": self._match_config(
                seed=seed,
                deck0=deck0,
                deck1=deck1,
                level_cap=11,
                minimum_card_level=11,
                king_tower_level=11,
            ),
            "options": {
                "render_mode": "native-render",
                "pause_native_render": True,
                "allow_initial_remaining_runtime_rejections": True,
                "decision_ticks": 10,
                "event_driven_decisions": False,
            },
        }
        try:
            self._battle.reset(**reset_kwargs)
        except Exception as error:  # noqa: BLE001 -- 引擎长会话退化：强制重启自愈
            print(
                f"FirstLight: reset failed ({str(error)[:120]}); restarting offline engine...",
                flush=True,
            )
            self._battle = None
            self._ensure_engine(restart=True)
            self._ensure_battle()
            self._battle.reset(**reset_kwargs)
        self._battle.native.set_speed(4.0)               # 热身期高速推进
        self._battle.native.resume()
        while True:
            obs0, _r, terminated, truncated, _i = self._battle.step_actor(
                0, {0: [self._action_v1.wait(0)]}
            )
            if obs0.tick >= FIRST_PLAYABLE_TICK or terminated or truncated:
                break
        self._battle.native.set_speed(self._speed)       # 游玩期速度（引擎保持运行）
        self._mask = self._battle.action_mask(0)
        native_now = self._battle.native.observe()
        image = self._capture()
        obs0_player = next(p for p in obs0.players if p.owner == 0)
        obs = {
            "image": image,
            "is_first": np.bool_(True),
            "is_last": np.bool_(False),
            "is_terminal": np.bool_(False),
            "elixir": np.array([self._probe_elixir(obs0_player, obs0.tick)], dtype=np.float32),
            "elixir_rate": np.array([self._elixir_rate(obs0.tick)], dtype=np.float32),
            "hand_ids": self._hand_ids(native_now),
            "deck_ids": np.array([DECK_INDEX[c] for c in self._deck0], dtype=np.int64),
            "action_mask": self._mask_flat(),
        }
        self._track_abilities(obs0)
        if self._opp_policy == "selfplay":
            self._mask1 = self._battle.action_mask(1)
            self._track_abilities_owner(obs0, 1)
            p1c = next(p for p in obs0.players if p.owner == 1)
            self._last_obs1 = {
                "image": image,
                "is_first": np.bool_(True),
                "is_last": np.bool_(False),
                "is_terminal": np.bool_(False),
                "elixir": np.array([float(np.clip(float(p1c.elixir_exact or 0.0), 0.0, 10.0))], dtype=np.float32),
                "elixir_rate": np.array([self._elixir_rate(obs0.tick)], dtype=np.float32),
                "hand_ids": self._hand_ids_owner(native_now, 1),
                "deck_ids": np.array([DECK_INDEX[c] for c in self._deck1], dtype=np.int64),
                "action_mask": self._mask_flat(self._mask1),
            }
        self._align_log_tick(obs0.tick, engine_elixir=float(next(p for p in obs0.players if p.owner == 0).elixir_exact or 0.0))
        self._last_obs = obs
        return obs

    def step(self, action, action1=None):
        """连续采集模式 step，带基础设施自愈。

        action1：owner-1 的扁平动作索引（M5 自博弈；None = 按opp_policy
        回退脚本）。结束后 self._last_obs1 持有 owner-1 视图观测。

        guest 侧游戏进程可能在长会话中崩溃/卡死（实测 ~24 分钟一例）：
        传输层一旦失败就重启引擎并以截断语义结束本局——环境绝不因引擎
        故障带崩训练，损失是丢弃当前 episode。
        """
        try:
            return self._step_inner(action, action1=action1)
        except (RunnerError, BattleEnvError, TimeoutError) as error:
            print(
                f"FirstLight: engine failure ({str(error)[:120]}); "
                "restarting engine, episode aborted",
                flush=True,
            )
            self._battle = None
            self._last_obs1 = None
            return self._abort_episode_obs()

    def _abort_episode_obs(self):
        if self._last_obs is not None:
            obs = dict(self._last_obs)
        else:
            obs = {
                "image": np.zeros(self._size + (3,), dtype=np.uint8),
                "elixir": np.zeros(1, dtype=np.float32),
                "elixir_rate": np.zeros(1, dtype=np.float32),
                "hand_ids": self._hand_ids(native_now),
                "action_mask": self._mask_flat(),
            }
        # 基础设施中止 = 截断（is_last 而非 is_terminal）：世界模型按可
        # bootstrap 的结局处理，不把引擎崩溃误学成"对局自然打完"。
        obs["is_last"] = np.bool_(True)
        obs["is_terminal"] = np.bool_(False)
        return obs, np.float32(0.0), True, {"tick": -1}

    def _step_inner(self, action, action1=None):
        """连续采集模式：引擎常跑不暂停，step 提交动作并推进一个决策周期。

        帧与动作的时序关系：返回的观测是动作执行后的战场（s_{t+1}），
        采集时刻的 tick 记录在 info["tick"]——引擎在两次采集之间会继续
        前进若干 tick（观测延迟），这是连续采集的固有语义，帧上已如实标注。
        """

        self._ensure_battle()
        slot, cx, cy = self._decode(action)
        legal = self._legal(slot, cx, cy)

        self._battle.native.resume()  # 若被外部暂停则恢复（常跑时无操作）
        # 一次 observe 服务本步所有手牌查询（player-0 card_id + 脚本对手），
        # 压缩提交关键路径的探针往返
        native_now = self._battle.native.observe()
        if self._opp_policy == "random":
            opp = self._scripted_action(native_now)
        elif self._opp_policy == "selfplay":
            opp = self._selfplay_action(action1, native_now)
        else:
            opp = None

        def with_opp(a0_list):
            d = {0: a0_list}
            if opp is not None:
                d[1] = [opp]
            return d

        if legal:
            if slot == -1:
                # 技能位：解析 (source_entity, ability_id) 提交 type-2 激活
                resolved = self._resolve_ability()
                if resolved is None:
                    # 快照过期等极端情形：降级 WAIT（仍走同一条提交通道）
                    play = self._action_v1.wait(0, ticks=10, execute_offset_ticks=3)
                else:
                    entity, ability_id = resolved
                    play = self._action_v1(
                        owner=0, kind="activate_ability", source_entity=entity,
                        ability_id=ability_id, next_decision_ticks=10,
                    )
            else:
                rp0 = next(p for p in native_now["players"] if p["owner"] == 0)
                hand_card = next(c for c in rp0["hand"] if c["handIndex"] == slot)
                # next_decision_ticks=10：固定 10 tick 决策节奏（2Hz@speed 1.0）——保证对局在
                # time_limit 内到达终局（±1 奖励才存在），并与 4Hz 协议一致
                play = self._action_v1.play(
                    owner=0, hand_slot=slot, grid=(cx, cy), card_id=hand_card["cardId"],
                    next_decision_ticks=10,
                )
            from dataclasses import replace

            last_error = None
            result = None
            for attempt in range(3):
                offset = 1 + attempt * 2
                try:
                    result = self._battle.step_actor(0, with_opp([replace(play, execute_offset_ticks=offset)]))
                    break
                except BattleEnvError as error:
                    # 决策死线竞态（偶发）：降级为等待。环境绝不因智能体动作而崩溃。
                    result = self._battle.step_actor(0, with_opp([self._action_v1.wait(0, ticks=10, execute_offset_ticks=3)]))
                    break
                except Exception as error:  # noqa: BLE001
                    last_error = error
                    if "not queued" not in str(error) or attempt == 2:
                        raise
                    self._battle.step_actor(0, {0: [self._action_v1.wait(0)]})
            obs0, reward, terminated, truncated, info = result
        else:
            obs0, reward, terminated, truncated, info = self._battle.step_actor(
                0, with_opp([self._action_v1.wait(0, ticks=10, execute_offset_ticks=3)])
            )

        self._track_abilities(obs0)
        if terminated or truncated:
            # 对局终局：引擎的回放播放器已停止 stepping，任何后续提交都会失败。
            # 自动重开下一局（ParallelEnv 惰性重置语义），终局奖励已在本步交付。
            self._needs_reset = True
        self._mask = self._battle.action_mask(0)
        self._mask1 = self._battle.action_mask(1) if self._opp_policy == "selfplay" else None
        reward = np.float32(reward)
        done = bool(terminated)
        image = self._capture()
        player = next(p for p in obs0.players if p.owner == 0)
        obs = {
            "image": image,
            "is_first": np.bool_(False),
            "is_last": np.bool_(done),
            "is_terminal": np.bool_(terminated),
            "elixir": np.array([self._probe_elixir(player, obs0.tick)], dtype=np.float32),
            "elixir_rate": np.array([self._elixir_rate(obs0.tick)], dtype=np.float32),
            "hand_ids": self._hand_ids(native_now),
            "deck_ids": np.array([DECK_INDEX[c] for c in self._deck0], dtype=np.int64),
            "action_mask": self._mask_flat(),
        }
        if self._mask1 is not None:
            self._track_abilities_owner(obs0, 1)
            p1c = next(p for p in obs0.players if p.owner == 1)
            obs["is_first"] = obs["is_first"]  # noqa: FURB122 -- keep dtype
            self._last_obs1 = {
                "image": image,
                "is_first": np.bool_(False),
                "is_last": np.bool_(done),
                "is_terminal": np.bool_(terminated),
                "elixir": np.array([float(np.clip(float(p1c.elixir_exact or 0.0), 0.0, 10.0))], dtype=np.float32),
                "elixir_rate": np.array([self._elixir_rate(obs0.tick)], dtype=np.float32),
                "hand_ids": self._hand_ids_owner(native_now, 1),
                "deck_ids": np.array([DECK_INDEX[c] for c in self._deck1], dtype=np.int64),
                "action_mask": self._mask_flat(self._mask1),
            }
        self._align_log_tick(obs0.tick, engine_elixir=float(next(p for p in obs0.players if p.owner == 0).elixir_exact or 0.0))
        self._last_obs = obs
        if done and self._episodes % 20 == 0:  # 每 20 局采样一次终局诊断
            try:
                diag = {k: str(v)[:80] for k, v in (info or {}).items()}
            except Exception:
                diag = str(info)[:200]
            print(f"FirstLight: ep end tick={obs0.tick} reward={reward} "
                  f"term={terminated} trunc={truncated} info={diag}", flush=True)
        return obs, reward, done, {"tick": int(obs0.tick)}

    def render(self):
        return self._capture()

    def close(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream = None
