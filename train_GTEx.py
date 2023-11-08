import argparse
import itertools
import random
from collections import defaultdict
from pathlib import Path

import gymnasium as gym
import gym_PBN
import numpy as np
import torch
from gym_PBN.utils.eval import compute_ssd_hist

import wandb
from bdq_model import BranchingDQN

from bdq_model.utils import ExperienceReplayMemory, AgentConfig

model_cls = BranchingDQN
model_name = "BranchingDQN"

# Parse settings
parser = argparse.ArgumentParser(description="Train an RL model for target control.")

parser.add_argument(
    "--resume-training",
    action="store_true",
    help="resume training from latest checkpoint.",
)
parser.add_argument("--checkpoint-dir", default="models", help="path to save models")
parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="disables CUDA training"
)
parser.add_argument(
    "--eval-only", action="store_true", default=False, help="evaluate only"
)

parser.add_argument(
    "--size", type=int, required=True, help="the experiment name."
)
parser.add_argument(
    "--exp-name", type=str, default="ddqn", metavar="E", help="the experiment name."
)
parser.add_argument("--env", type=str, help="the environment to run.")

parser.add_argument("--log-dir", default="logs", help="path to save logs")

args = parser.parse_args()

# # Load env
env = gym.make(f"gym-PBN/GTEx")

# set up logs
TOP_LEVEL_LOG_DIR = Path(args.log_dir)
TOP_LEVEL_LOG_DIR.mkdir(parents=True, exist_ok=True)

RUN_NAME = f"{args.env}_pbn{args.size}_{args.exp_name}"

# Checkpoints
checkpoint_path = Path(args.checkpoint_dir) / RUN_NAME
checkpoint_path.mkdir(parents=True, exist_ok=True)


def get_latest_checkpoint():
    files = list(checkpoint_path.glob("*.pt"))
    if len(files) > 0:
        return max(files, key=lambda x: x.stat().st_ctime)
    else:
        return None

def state_equals(state1, state2):
    for i in range(len(state2)):
        if state1[i] != state2[i]:
            return False
    return True


config = AgentConfig()

state_len = env.observation_space.shape[0]
model = BranchingDQN((state_len, state_len), state_len + 1, config, env)
model.to(device=model.config.device)

# config = model.get_config()
# config["learning_starts"] = args.learning_starts
run = wandb.init(
    project="pbn-rl",
    sync_tensorboard=True,
    monitor_gym=True,
    config={},
    name=RUN_NAME,
    save_code=True,
)

print(checkpoint_path)

model.learn(
    env=env,
    path=checkpoint_path,
    wandb=run
)

attrs = env.all_attractors
print(f"final attractors were ({len(env.all_attractors)}")
for attr in env.all_attractors:
    print(attr)

print("skip testig the model")

env.close()
run.finish()
