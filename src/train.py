from argparse import ArgumentParser
from os import makedirs
from os.path import join

import ray
import wandb
from omegaconf import OmegaConf
from ray import tune
from ray.rllib.agents import ppo
from tqdm.auto import trange
from wandb import Video

from custom_dungeon import CustomDungeon
from evaluate import evaluate
from utils import build_ppo_config, fix_everything


def train(config_path: str):
    config = OmegaConf.load(config_path)
    config = OmegaConf.to_container(config, resolve=True)

    fix_everything(config["seed"])

    mode = "offline" if config.get("log_offline", False) else "online"
    wandb_run = wandb.init(project="RL-vacuum", config=config, mode=mode)

    ray.shutdown()
    ray.init(ignore_reinit_error=True)
    tune.register_env("Dungeon", lambda it: CustomDungeon(**it))

    train_config = build_ppo_config(config)

    checkpoint_dir = join(wandb_run.dir, config["checkpoint_dir"])
    makedirs(checkpoint_dir)
    result_dir = join(wandb_run.dir, config["result_dir"])
    makedirs(result_dir)

    agent = ppo.PPOTrainer(train_config)

    epoch_loop = trange(config["n_iterations"], desc="Training")
    for i in epoch_loop:
        result = agent.train()

        if (i + 1) % config["checkpoint_step"] == 0:
            agent.save(checkpoint_dir)

        wandb_run.log(
            {
                "train/reward_min": result["episode_reward_min"],
                "train/reward_mean": result["episode_reward_mean"],
                "train/reward_max": result["episode_reward_max"],
                "train/mean_length": result["episode_len_mean"],
            },
            step=i,
        )

        if (i + 1) % config["eval_step"] == 0:
            result_gif = evaluate(agent, config, join(result_dir, f"step-{i + 1}"))
            wandb_run.log({"val/gif": Video(result_gif, fps=30, format="gif")}, step=i)

    epoch_loop.close()


if __name__ == "__main__":
    __arg_parser = ArgumentParser()
    __arg_parser.add_argument("config", help="Path to YAML config")

    __args = __arg_parser.parse_args()
    train(__args.config)
