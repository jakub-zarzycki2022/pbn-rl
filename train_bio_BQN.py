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

args = parser.parse_args()

# Load env
# A systems described by biologist we are cooperating with
# https://docs.google.com/document/d/1ACSjckbhof64rzLWtEUVE0lTKJuVde7dfFSTxNbii3o/edit
env = gym.make(f"gym-PBN/PBNEnv",
               N=args.size,
               genes=[
                "Pax7", "Myf5", "MyoD1", "MyoG", "miR1", "miR206", "FGF8", "SHH",
                "Pax3", "Mrf4", "Mef2c", "Mef2a", "ID3", "WNT", "WNT3a", "T", "Msg1",
                "twist", "pitx2", "mir133a1", "notch", "meox1", "creb1", "wwtr1",
                "myh10", "myh1", "myh2", "msgn1"
               ],
               logic_functions=[
                    [('(not miR1 and not MyoG and not miR206) and (T or notch)', 1.0)],  # pax7
                    [('Pax7 or Pax3 or WNT or SHH', 1.0)],  # myf5
                    [('(not ID3 or not notch or not Pax7) and (FGF8 or Mef2c or Mef2a or SHH or WNT or Pax3 or twist or pitx2)', 1.0)],  # myod1
                    [('not notch and (MyoG or MyoD1 or twist)', 1.0)],  # myog
                    [('Myf5', 1.0)],  # mir1
                    [('MyoG or Myf5 or MyoD1 or Mef2c', 1.0)],  # mir206
                    [('FGF8', 1.0)],    # fgf8(in)
                    [('SHH', 1.0)],     # shh(in)
                    [('(not miR1 and not miR206) and (meox1 or creb1 or wwtr1)', 1.0)],    # pax3
                    [('MyoG or Mef2c or Mef2a', 1.0)],  # mrf4 (myf6)
                    [('not Mrf4 and twist', 1.0)],   # mef2c
                    [('twist', 1.0)],           # mef2a
                    [('ID3', 1.0)],             # id3(in)
                    [('WNT', 1.0)],             # wnt(in)
                    [('WNT3a', 1.0)],           # wnt3a(in)
                    [('WNT3a or FGF8', 1.0)],   # t
                    [('WNT3a', 1.0)],           # msg1
                    [("WNT", 1.0)],             # twist
                    [("Pax3", 1.0)],            # pitx2
                    [("MyoG", 1.0)],            # mir133a1
                    [("notch", 1.0)],           # notch
                    [("Pax3", 1.0)],            # meox1
                    [("creb1", 1.0)],           # creb1
                    [("wwtr1", 1.0)],           # wwtr1
                    [("Mef2a", 1.0)],           # myh10
                    [("Mef2a", 1.0)],           # myh1
                    [("Mef2a", 1.0)],           # myh2
                    [("T", 1.0)]                # msgn1
               ])
print(type(env.env.env))
print("#attractors = ", len(env.all_attractors))
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
