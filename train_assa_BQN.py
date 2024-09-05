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
from gbdq_model import GBDQ

from gbdq_model.utils import ExperienceReplayMemory, AgentConfig

model_cls = GBDQ
model_name = "GBDQ"

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

parser.add_argument("--assa-file", type=str, required=True)

args = parser.parse_args()

with open(args.assa_file, "r") as env_file:
    genes = []
    logic_funcs = defaultdict(list)

    for line in env_file:
        print(line, end="")
        line = line.split()

        if len(line) == 0:
            continue

        # get all vars
        if line[0] == "Vars:":
            while True:
                line = next(env_file)
                line = line.split()

                if line[0] == "end":
                    break

                if line[0][-1] == ":":
                    genes.append(line[0][:-1])
                else:
                    genes.append(line[0])

        if line[0] == "Evolution:":
            while True:
                line = next(env_file)
                line = line.split()

                if len(line) == 0:
                    continue

                if line[0] == "end":
                    break

                target_gene = line[0].split("=")[0]
                if line[0].split("=")[1] == "false":
                    continue

                for i in range(len(line)):
                    sline = line[i].split("=")
                    if sline[-1] == "false":
                        line[i] = f"( not {sline[0]} )"
                    else:
                        line[i] = sline[0]

                if target_gene == "EGFR":
                    target_fun = "True"
                    logic_funcs[target_gene].append((target_fun, 1.0))
                    continue

                target_fun = " ".join(line[2:])
                target_fun = target_fun.replace("(", " ( ")
                target_fun = target_fun.replace(")", " ) ")
                target_fun = target_fun.replace("|", " or ")
                target_fun = target_fun.replace("&", " and ")
                target_fun = target_fun.replace("~", " not ")
                logic_funcs[target_gene].append((target_fun, 1.0))

print(list(logic_funcs.keys()))
print(list(logic_funcs.values()))


for i in range(len(genes)):
    print(list(logic_funcs.keys())[i], list(logic_funcs.values())[i])

print(logic_funcs)

# Load env
env = gym.make(f"gym-PBN/PBNEnv",
               N=args.size,
               genes=list(logic_funcs.keys()),
               logic_functions=list(logic_funcs.values()))

print(env)
# raise ValueError
print(type(env.env.env))
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
        if state1[i]  != state2[i]:
            return False
    return True


config = AgentConfig()

state_len = env.observation_space.shape[0]
model = model_cls(state_len, state_len + 1, config, env)
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
print(f"final pseudo0attractors were ({len(env.all_attractors)})")
print(f"final real attractors were ({len(env.real_attractors)})")
pseudo = set([i[0] for i in env.all_attractors])
real = set(i[0] for i in env.real_attractors)
print(f"intersection size: {len(pseudo.intersection(real))}")

print("skip testig the model")

env.close()
run.finish()
