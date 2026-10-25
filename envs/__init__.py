import atexit

from . import parallel, wrappers


def make_envs(config):
    suite = config.task.split("_", 1)[0]

    if suite == "isaaclab":
        return _make_isaaclab_envs(config)

    if suite == "firstlight":
        # 进程内适配器：绕开 ParallelEnv 的多进程管道（长会话下主/子进程
        # 互等死锁）。多引擎 = 每 env 一个 MuMu 实例（serials/control_ports
        # 按 env 索引给出），线程池并行步进，socket 等待释放 GIL 真重叠。
        from envs.firstlight import FirstLightInlineEnv

        env_num = int(getattr(config, "env_num", 1) or 1)

        def firstlight_constructor(idx):
            return make_env(config, idx)

        env = FirstLightInlineEnv(firstlight_constructor, device=config.device, env_num=env_num)
        return env, None, env.observation_space, env.action_space

    def env_constructor(idx):
        return lambda: make_env(config, idx)

    train_envs = parallel.ParallelEnv(env_constructor, config.env_num, config.device)
    eval_envs = (
        parallel.ParallelEnv(env_constructor, config.eval_episode_num, config.device)
        if config.eval_episode_num > 0
        else None
    )
    obs_space = train_envs.observation_space
    act_space = train_envs.action_space
    return train_envs, eval_envs, obs_space, act_space


def _make_isaaclab_envs(config):
    if int(config.eval_episode_num) > 0:
        raise ValueError(
            "IsaacLab environments do not support separate eval envs yet. Set eval_episode_num: 0 in the env config."
        )

    # AppLauncher must be created before any isaaclab.* module is imported.
    from isaaclab.app import AppLauncher

    headless = getattr(config, "headless", True)
    launcher = AppLauncher(headless=headless, enable_cameras=True)
    sim_app = launcher.app

    from envs.isaaclab import IsaacLabVecEnv, create_isaaclab_env

    raw_env = create_isaaclab_env(config)

    train_envs = IsaacLabVecEnv(raw_env, simulation_app=sim_app, image_size=tuple(config.size))

    # Ensure the SimulationApp is shut down cleanly on process exit.
    atexit.register(train_envs.close)

    eval_envs = None  # Not supported yet for IsaacLab.

    obs_space = train_envs.observation_space
    act_space = train_envs.action_space
    return train_envs, eval_envs, obs_space, act_space


def make_env(config, id):
    suite, task = config.task.split("_", 1)
    if suite == "dmc":
        import envs.dmc as dmc

        env = dmc.DeepMindControl(task, config.action_repeat, config.size, seed=config.seed + id)
        env = wrappers.NormalizeActions(env)
    elif suite == "atari":
        import envs.atari as atari

        env = atari.Atari(
            task,
            config.action_repeat,
            config.size,
            gray=config.gray,
            noops=config.noops,
            lives=config.lives,
            sticky=config.sticky,
            actions=config.actions,
            length=config.time_limit,
            pooling=config.pooling,
            aggregate=config.aggregate,
            resize=config.resize,
            autostart=config.autostart,
            clip_reward=config.clip_reward,
            seed=config.seed + id,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "memorymaze":
        from envs.memorymaze import MemoryMaze

        env = MemoryMaze(task, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "crafter":
        import envs.crafter as crafter

        env = crafter.Crafter(task, config.size, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "firstlight":
        import envs.firstlight as firstlight

        serials = list(config.get("serials", []) or [])
        control_ports = list(config.get("control_ports", []) or [])
        env = firstlight.FirstLight(
            task,
            config.size,
            seed=config.seed + id,
            speed=config.speed,
            capture_backend=config.get("capture", "adb"),
            deck_mode=config.get("deck_mode", "fixed"),
            opp_policy=config.get("opp_policy", "afk"),
            vm_index=(config.get("vm_indexes", []) or [None] * 8)[id] if id < len(config.get("vm_indexes", []) or []) else None,
            vm_name=(config.get("vm_names", []) or [None] * 8)[id] if id < len(config.get("vm_names", []) or []) else None,
            serial=serials[id] if id < len(serials) else None,
            control_port=control_ports[id] if id < len(control_ports) else None,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "metaworld":
        import envs.metaworld as metaworld

        env = metaworld.MetaWorld(
            task,
            config.action_repeat,
            config.size,
            config.camera,
            config.seed + id,
        )
    else:
        raise NotImplementedError(suite)
    env = wrappers.TimeLimit(env, config.time_limit // config.action_repeat)
    return wrappers.Dtype(env)
