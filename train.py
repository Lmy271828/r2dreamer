import atexit
import pathlib
import sys
import warnings

import hydra
import torch

import tools
from buffer import Buffer
from dreamer import Dreamer
from envs import make_envs
from trainer import OnlineTrainer

warnings.filterwarnings("ignore")
sys.path.append(str(pathlib.Path(__file__).parent))
# torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


@hydra.main(version_base=None, config_path="configs", config_name="configs")
def main(config):
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)

    # Mirror stdout/stderr to a file under logdir while keeping console output.
    console_f = tools.setup_console_log(logdir, filename="console.log")
    atexit.register(lambda: console_f.close())

    print("Logdir", logdir)

    logger = tools.Logger(
        logdir,
        wandb_config=config.wandb,
        run_config=__import__("omegaconf", fromlist=["OmegaConf"]).OmegaConf.to_container(config, resolve=True),
    )
    # save config
    logger.log_hydra_config(config)

    replay_buffer = Buffer(config.buffer)

    print("Create envs.")
    train_envs, eval_envs, obs_space, act_space = make_envs(config.env)

    print("Simulate agent.")
    agent = Dreamer(
        config.model,
        obs_space,
        act_space,
    ).to(config.device)

    # 断点续训（权重+optimizer）：+resume_ckpt=<latest.pt 路径>。
    # replay buffer 不持久化（内存态），恢复后从空 buffer 重新采集，
    # learner 会等 batch_length 帧回填后才恢复更新——权重热、数据冷。
    resume_ckpt = str(config.get("resume_ckpt", "") or "")
    if resume_ckpt:
        blob = torch.load(resume_ckpt, map_location=config.device, weights_only=False)
        agent.load_state_dict(blob["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, blob["optims_state_dict"])
        print(f"Resumed weights+optim from {resume_ckpt} (replay buffer restarts empty)", flush=True)

    policy_trainer = OnlineTrainer(config.trainer, replay_buffer, logger, logdir, train_envs, eval_envs)
    policy_trainer.resume_ckpt = resume_ckpt  # 供 trainer 恢复联赛成熟态（league_state.pt）
    policy_trainer.begin(agent)

    items_to_save = {
        "agent_state_dict": agent.state_dict(),
        "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
    }
    torch.save(items_to_save, logdir / "latest.pt")


if __name__ == "__main__":
    main()
