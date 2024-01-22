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
env = gym.make(f"gym-PBN/PBNEnv",
               N=args.size,
               genes=['v_AP', 'v_AgAb', 'v_BC', 'v_Bb', 'v_C', 'v_DCI', 'v_DCII', 'v_DP', 'v_EC', 'v_IFNgI',
                      'v_IFNgII', 'v_IL10I', 'v_IL10II', 'v_IL12I', 'v_IL12II', 'v_IL4I', 'v_IL4II', 'v_IgA',
                      'v_IgG', 'v_MPI', 'v_NE', 'v_Oag', 'v_PH', 'v_PIC', 'v_T0', 'v_TTSSI', 'v_TTSSII', 'v_Th1I',
                      'v_Th1II', 'v_Th2I', 'v_Th2II', 'v_TrI', 'v_TrII'],
               logic_functions=[[('((v_IgG and ((v_C and v_Bb) and (v_MPI and v_Th1I))) or (v_AgAb and ((v_MPI and v_Th1I) and v_Bb)))', 1.0)],
                                [('((v_IgA and v_Bb) or (v_IgG and v_Bb))', 1.0)],
                                [('(v_T0 or v_BC)', 1.0)],
                                [('(v_Bb and not v_PH)', 1.0)],
                                [('((v_Bb and not v_Oag) or (v_IgG and v_AgAb))', 1.0)],
                                [('((v_IFNgI and v_Bb) or (v_PIC and v_Bb))', 1.0)],
                                [('v_DCI', 1.0)],
                                [('(v_NE and v_TTSSI)', 1.0)],
                                [('v_Bb', 1.0)],
                                [('(((v_DCI and not v_IL4I) or (v_MPI and not v_IL4I)) or (v_Th1I and not (v_IL10I or v_IL4I)))', 1.0)],
                                [('v_IFNgI', 1.0)],
                                [('(((v_Th2I and v_TTSSI) or v_TrI) or v_MPI)', 1.0)],
                                [('v_IL10I', 1.0)],
                                [('((v_DCII and v_T0) and not v_IL4II)', 1.0)],
                                [('((v_DCII and v_T0) and not v_IL4II)', 1.0)],
                                [('v_IL4II', 1.0)],
                                [('((v_Th2II and not (v_IL12II or v_IFNgII)) or ((v_DCII and v_T0) and not (v_IL12II or v_IFNgII)))', 1.0)],
                                [('((v_IgA and v_Bb) or (v_BC and v_Bb))', 1.0)],
                                [('(v_IgG or v_BC)', 1.0)],
                                [('((v_IFNgI and v_Bb) or (v_PIC and v_Bb))', 1.0)],
                                [('v_PIC', 1.0)],
                                [('v_Bb', 1.0)],
                                [('(v_AP and v_Bb)', 1.0)],
                                [('(((v_DP and not v_IL10I) or (v_EC and not v_IL10I)) or (v_AP and not v_IL10I))', 1.0)],
                                [('v_DCII', 1.0)],
                                [('(v_Bb and not (v_IgA or v_IgG))', 1.0)],
                                [('v_TTSSI', 1.0)],
                                [('v_Th1II', 1.0)],
                                [('(v_DCII and (v_IL12II and v_T0))', 1.0)],
                                [('v_Th2II', 1.0)],
                                [('((v_DCII and v_T0) and not v_IL12II)', 1.0)],
                                [('v_TrII', 1.0)],
                                [('(v_DCII and v_T0)', 1.0)]])

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
print(f"final pseudo0attractors were ({len(env.all_attractors)})")
print(f"final real attractors were ({len(env.real_attractors)})")
pseudo = set([i[0] for i in env.all_attractors])
real = set(i[0] for i in env.real_attractors)
print(f"intersection size: {len(pseudo.intersection(real))}")

print("skip testig the model")

env.close()
run.finish()
