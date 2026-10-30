import copy
import threading
import time

import torch

import tools


class OnlineTrainer:
    def __init__(self, config, replay_buffer, logger, logdir, train_envs, eval_envs):
        self.replay_buffer = replay_buffer
        self.logger = logger
        self.train_envs = train_envs
        self.eval_envs = eval_envs
        self.steps = int(config.steps)
        self.pretrain = int(config.pretrain)
        self.eval_every = int(config.eval_every)
        self.eval_episode_num = int(config.eval_episode_num)
        self.video_pred_log = bool(config.video_pred_log)
        self.params_hist_log = bool(config.params_hist_log)
        self.batch_length = int(config.batch_length)
        batch_steps = int(config.batch_size * config.batch_length)
        # train_ratio is based on data steps rather than environment steps.
        self._updates_needed = tools.Every(batch_steps / config.train_ratio * config.action_repeat)
        self._should_pretrain = tools.Once()
        self._should_log = tools.Every(config.update_log_every)
        self._should_eval = tools.Every(self.eval_every)
        self._action_repeat = config.action_repeat

    def eval(self, agent, train_step):
        """Run evaluation episodes.

        For CPU-based environments (``ParallelEnv``), stepping is executed on
        CPU and observations are moved to GPU asynchronously.  For GPU-resident
        environments (``IsaacLabVecEnv``), no device transfer is needed —
        ``.to()`` is a no-op when source and target devices match.
        """
        print("Evaluating the policy...")
        envs = self.eval_envs
        agent.eval()
        # (B,)
        done = torch.ones(envs.env_num, dtype=torch.bool, device=agent.device)
        once_done = torch.zeros(envs.env_num, dtype=torch.bool, device=agent.device)
        steps = torch.zeros(envs.env_num, dtype=torch.int32, device=agent.device)
        returns = torch.zeros(envs.env_num, dtype=torch.float32, device=agent.device)
        log_metrics = {}
        # cache is only used for video logging / open-loop prediction.
        cache = []
        agent_state = agent.get_initial_state(envs.env_num)
        # (B, A)
        act = agent_state["prev_action"].clone()
        while not once_done.all():
            steps += ~done * ~once_done
            # Step environments.  Each env backend handles device placement
            # internally (ParallelEnv converts to CPU, IsaacLabVecEnv keeps
            # on GPU).  The .to() calls below are no-ops when the data is
            # already on agent.device.
            # (B, A), (B,)
            trans, step_done = envs.step(act.detach(), done)
            # dict of (B, 1, *)
            trans = trans.to(agent.device, non_blocking=True)
            # (B,)
            done = step_done.to(agent.device)

            # Store transition.
            # We keep the observation and the action that produced it together.
            trans["action"] = act
            if len(cache) < self.batch_length:
                cache.append(trans.clone())
            # (B, A)
            act, agent_state = agent.act(trans, agent_state, eval=True)
            returns += trans["reward"][:, 0] * ~once_done
            for key, value in trans.items():
                if key.startswith("log_"):
                    if key not in log_metrics:
                        log_metrics[key] = torch.zeros_like(returns)
                    log_metrics[key] += value[:, 0] * ~once_done
            once_done |= done
        # dict of (B, T, *)
        cache = torch.stack(cache, dim=1) if len(cache) else None
        self.logger.scalar("episode/eval_score", returns.mean())
        self.logger.scalar("episode/eval_length", steps.to(torch.float32).mean())
        for key, value in log_metrics.items():
            if key == "log_success":
                value = torch.clip(value, max=1.0)  # make sure 1.0 for success episode
            self.logger.scalar(f"episode/eval_{key[4:]}", value.mean())
        if cache is not None and "image" in cache:
            self.logger.video("eval_video", tools.to_np(cache["image"][:1]))
        if self.video_pred_log and cache is not None:
            initial = agent.get_initial_state(1)
            self.logger.video(
                "eval_open_loop",
                tools.to_np(
                    agent.video_pred(
                        cache[:1],  # give only first batch
                        (initial["stoch"], initial["deter"]),
                    )
                ),
            )
        self.logger.write(train_step)
        agent.train()

    def begin(self, agent):
        """Main online training loop.

        For CPU-based environments the loop overlaps CPU stepping and GPU
        model execution via pinned-memory async H2D transfers.  For
        GPU-resident environments (IsaacLab) no transfer is needed —
        ``.to()`` is a no-op when the data is already on the target device.
        """
        envs = self.train_envs
        # M5 自博弈（P3 定案）：对手 = 主策略的周期性冻结快照；热身期前
        # act1=None（env 回退随机脚本），warmup 后 deepcopy 快照接管，
        # 每 refresh_episodes 局用主策略权重刷新一次。
        selfplay = getattr(envs, "selfplay", False)
        self._opp_agent = None
        self._opp_state = None
        self._opp_episodes = 0
        self._opp_refresh_at = None
        if selfplay:
            print(
                f"M5: selfplay enabled (warmup={getattr(envs, 'selfplay_warmup', 40)} episodes, "
                f"refresh={getattr(envs, 'selfplay_refresh', 30)} episodes)",
                flush=True,
            )
        video_cache = []
        step = self.replay_buffer.count() * self._action_repeat
        # (B,)
        done = torch.ones(envs.env_num, dtype=torch.bool, device=agent.device)
        returns = torch.zeros(envs.env_num, dtype=torch.float32, device=agent.device)
        lengths = torch.zeros(envs.env_num, dtype=torch.int32, device=agent.device)
        episode_counter = int(getattr(envs, "start_episode_index", 0))
        win_history = []  # M5：主策略对快照胜率的滚动窗口（对脚本对手时语义为对脚本胜率）
        episode_ids = torch.arange(
            envs.env_num, dtype=torch.int32, device=agent.device
        )  # Kept constant so short episodes (< batch_length) remain sampable; RSSM resets via is_first.
        train_metrics = tools.MetricAccumulator()
        agent_state = agent.get_initial_state(envs.env_num)
        # (B, A)
        act = agent_state["prev_action"].clone()

        # Actor-learner 分离：更新在后台线程执行，采集循环不再被 1-2 秒的
        # update 阻塞（实测占空 ~29%）。同设备默认流上内核按提交序串行，
        # act 与 update 并发只造成"内核粒度的新旧权重混读"，单步增量在
        # 1e-6 lr 量级下无害；buffer 的 add/sample 并发同理属有界噪声。
        self._learner_stop = threading.Event()
        shared = {"step": step, "update_count": 0}
        self._metrics_lock = threading.Lock()

        def learner():
            local_updates = 0
            pretrain_used = False
            try:
                while not self._learner_stop.is_set() and shared["step"] < self.steps:
                    step_now = shared["step"]
                    if step_now // (envs.env_num * self._action_repeat) <= self.batch_length + 1:
                        time.sleep(0.05)
                        continue
                    if not pretrain_used:
                        update_num = self.pretrain
                        pretrain_used = True
                    else:
                        update_num = int(self._updates_needed(step_now))
                    for _ in range(update_num):
                        if self._learner_stop.is_set() or shared["step"] >= self.steps:
                            break
                        mets = agent.update(self.replay_buffer)
                        with self._metrics_lock:
                            train_metrics.add(mets)
                        local_updates += 1
                        shared["update_count"] = local_updates
                        if local_updates % 64 == 0:  # ~7 分钟一次的崩溃保险
                            torch.save(
                                {"agent_state_dict": agent.state_dict(),
                                 "optims_state_dict": tools.recursively_collect_optim_state_dict(agent)},
                                self.logger._logdir / "latest.pt")
                    if update_num == 0:
                        time.sleep(0.02)
            except Exception:  # noqa: BLE001 -- 学习线程死亡绝不能静默
                import traceback

                traceback.print_exc()
                print(
                    "learner thread crashed; collection continues without updates "
                    "(watch train/opt/updates freeze on wandb)",
                    flush=True,
                )

        learner_thread = threading.Thread(target=learner, daemon=True, name="learner")
        learner_thread.start()

        # fps 拆账：分相计时（每 1000 步落一次 p50）
        phase_ms = {"envs_step": [], "act": [], "buffer_add": [], "misc": []}

        while step < self.steps:
            # Evaluation
            if self._should_eval(step) and self.eval_episode_num > 0 and self.eval_envs is not None:
                self.eval(agent, step)
            # Save metrics
            if done.any():
                for i, d in enumerate(done):
                    if d and lengths[i] > 0:
                        if i == 0 and len(video_cache) > 0:
                            video = torch.stack(video_cache, axis=0)
                            self.logger.video("train_video", tools.to_np(video[None]))
                            video_cache = []
                        self.logger.scalar("episode/score", returns[i])
                        self.logger.scalar("episode/length", lengths[i])
                        episode_counter += 1
                        win_history.append(1.0 if returns[i] > 0 else 0.0)
                        if len(win_history) > 15:
                            win_history.pop(0)
                        self.logger.scalar("episode/index", episode_counter)  # 全局第几局（含热身）；wandb 面板 x 轴选它，曲线本身可隐藏
                        self.logger.scalar("episode/win_rate", sum(win_history) / len(win_history))  # 主策略平均胜率（滚动 15 局）
                        self.logger.write(step + i)  # to show all values on tensorboard
                        returns[i] = lengths[i] = 0
            step += int((~done).sum()) * self._action_repeat  # step is based on env side
            lengths += ~done

            # Step environments.  Each env backend handles device placement
            # internally (ParallelEnv converts to CPU, IsaacLabVecEnv keeps
            # on GPU).  The .to() calls below are no-ops when the data is
            # already on agent.device.
            # (B, A), (B,)
            act1 = None
            if selfplay:
                warmup = getattr(envs, "selfplay_warmup", 40)
                refresh = getattr(envs, "selfplay_refresh", 30)
                if self._opp_agent is None and self._opp_episodes >= warmup:
                    self._opp_agent = copy.deepcopy(agent)
                    self._opp_agent.eval()
                    self._opp_state = self._opp_agent.get_initial_state(envs.env_num)
                    self._opp_refresh_at = self._opp_episodes + refresh
                    print(f"M5: selfplay opponent activated after {self._opp_episodes} episodes", flush=True)
                elif self._opp_agent is not None and self._opp_episodes >= self._opp_refresh_at:
                    self._opp_agent.load_state_dict(agent.state_dict())
                    self._opp_refresh_at = self._opp_episodes + refresh
                    print(f"M5: opponent snapshot refreshed at {self._opp_episodes} episodes", flush=True)
                if self._opp_agent is not None:
                    obs1 = envs.pop_obs1()
                    if obs1 is not None:
                        obs1 = obs1.to(agent.device, non_blocking=True)
                        with torch.no_grad():
                            a1, self._opp_state = self._opp_agent.act(obs1, self._opp_state, eval=True)
                        act1 = a1
                else:
                    envs.pop_obs1()  # 热身期丢弃，避免陈旧观测跨期
            _t = time.perf_counter()
            trans, step_done = envs.step(act.detach(), done, action1=act1)
            phase_ms["envs_step"].append((time.perf_counter() - _t) * 1000)
            # dict of (B, 1, *)
            trans = trans.to(agent.device, non_blocking=True)
            # (B,)
            done = step_done.to(agent.device)
            if selfplay:
                self._opp_episodes += int(done.sum())

            # Policy inference on GPU.
            # "agent_state" is reset by the agent based on the "is_first" flag in trans.
            # (B, A)
            _t = time.perf_counter()
            act, agent_state = agent.act(trans.clone(), agent_state, eval=False)
            phase_ms["act"].append((time.perf_counter() - _t) * 1000)

            # Store transition.
            # We keep the observation and the action that produced it together.
            # Mask actions after an episode has ended.
            trans["action"] = act * ~done.unsqueeze(-1)
            trans["stoch"] = agent_state["stoch"]
            trans["deter"] = agent_state["deter"]
            trans["episode"] = episode_ids  # Don't lift dim
            if "image" in trans:
                video_cache.append(trans["image"][0])
            _t = time.perf_counter()
            self.replay_buffer.add_transition(trans.detach())
            phase_ms["buffer_add"].append((time.perf_counter() - _t) * 1000)
            returns += trans["reward"][:, 0]
            # Actor-learner 分离：更新由后台 learner 线程执行（见上），采集循环零阻塞
            shared["step"] = step
            # Log training metrics（指标由 learner 线程在锁内积累）
            if self._should_log(step):
                import numpy as _np
                for pk, pv in phase_ms.items():
                    if pv:
                        self.logger.scalar(f"train/timing/{pk}_p50", float(_np.percentile(pv, 50)))
                        self.logger.scalar(f"train/timing/{pk}_p90", float(_np.percentile(pv, 90)))
                        phase_ms[pk] = pv[-500:]
                _step_total = sum(_np.percentile(pv, 50) for pv in phase_ms.values() if pv)
                self.logger.scalar("train/timing/envs_step_share",
                                   _np.percentile(phase_ms["envs_step"], 50) / max(1e-9, _step_total)
                                   if phase_ms["envs_step"] else 0.0)
                with self._metrics_lock:
                    mets = train_metrics.mean()
                for name, value in mets.items():
                    self.logger.scalar(f"train/{name}", value)
                self.logger.scalar("train/opt/updates", shared["update_count"])
                if self.video_pred_log:
                    data, _, initial = self.replay_buffer.sample(prefer_online=False)
                    self.logger.video("open_loop", tools.to_np(agent.video_pred(data, initial)))
                if self.params_hist_log:
                    for name, param in agent._named_params.items():
                        self.logger.histogram(name, tools.to_np(param))
                self.logger.write(step, fps=True)

        # 步数到点：停 learner、收尾排干指标
        self._learner_stop.set()
        learner_thread.join(timeout=300)
        with self._metrics_lock:
            final_mets = train_metrics.mean()
        for name, value in final_mets.items():
            self.logger.scalar(f"train/{name}", value)
        self.logger.scalar("train/opt/updates", shared["update_count"])
        self.logger.write(step, fps=True)
