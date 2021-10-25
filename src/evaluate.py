from argparse import ArgumentParser
from os import makedirs
from os.path import join
from typing import Dict

import ray
from omegaconf import OmegaConf
from ray import tune
from ray.rllib.agents import ppo
from tqdm.auto import trange

from custom_dungeon import CustomDungeon
from src.utils import build_ppo_config


def evaluate(agent, config: Dict, output_dir: str) -> str:
    env = CustomDungeon(**config["ppo"]["env_config"])
    if "seed" in config:
        env.seed(config["seed"])
    obs = env.reset()

    makedirs(output_dir, exist_ok=True)
    env.get_image_view().save(join(output_dir, "start.png"))

    frames = []

    for _ in range(500):
        action = agent.compute_single_action(obs)

        frame = env.get_image_view().quantize()
        frames.append(frame)

        # frame.save('tmp1.png')
        obs, reward, done, info = env.step(action)
        if done:
            break

    result_gif = join(output_dir, "results.gif")
    frames[0].save(result_gif, save_all=True, append_images=frames[1:], loop=0, duration=1000 / 60)
    return result_gif


def run_evaluation(config_path: str, checkpoint_path: str, output_path: str, n_samples: int = 5):
    config = OmegaConf.load(config_path)
    config = OmegaConf.to_container(config, resolve=True)

    # To see different environment samples
    del config["seed"]

    ray.shutdown()
    ray.init(ignore_reinit_error=True)
    tune.register_env("Dungeon", lambda it: CustomDungeon(**it))

    train_config = build_ppo_config(config)
    agent = ppo.PPOTrainer(train_config)
    agent.restore(checkpoint_path)

    for i in trange(n_samples, desc="Evaluation"):
        evaluate(agent, config, join(output_path, f"run-{i + 1}"))


if __name__ == "__main__":
    __arg_parser = ArgumentParser()
    __arg_parser.add_argument("-c", "--config", required=True, help="Path to YAML config")
    __arg_parser.add_argument("-m", "--model", required=True, help="Path to model's checkpoint")
    __arg_parser.add_argument("-o", "--output", required=True, help="Path to output folder")
    __arg_parser.add_argument("--n-samples", type=int, default=5, help="Number of samples to generate")

    __args = __arg_parser.parse_args()
    run_evaluation(__args.config, __args.model, __args.output, __args.n_samples)
