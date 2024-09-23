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
               N=28,
               genes=["v_ADD", "v_ATM", "v_ATR", "v_BRCA1", "v_CHK1", "v_CHK2", "v_CHKREC", "v_DNAPK", "v_DSB",
                      "v_FAN1", "v_FANCD1N", "v_FANCD2I", "v_FANCJBRCA1", "v_FANCM", "v_FAcore", "v_H2AX", "v_HRR",
                      "v_ICL", "v_KU", "v_MRN", "v_MUS81", "v_NHEJ", "v_PCNATLS", "v_RAD51", "v_USP1", "v_XPF", "v_p53",
                      "v_ssDNARPA"],
               logic_functions=[[("((v_ADD and not v_PCNATLS) or ((v_MUS81 and (v_FAN1 or v_XPF)) and not v_PCNATLS))", 1.0)],
                                 [("((v_DSB and not v_CHKREC) or (v_ATR and not v_CHKREC))", 1.0)],
                                 [("(((v_FANCM and not v_CHKREC) or (v_ATM and not v_CHKREC)) or (v_ssDNARPA and not v_CHKREC))", 1.0)],
                                 [("((v_DSB and ((v_ATM or v_CHK2) or v_ATR)) and not v_CHKREC)", 1.0)],
                                 [("(((v_ATM and not v_CHKREC) or (v_DNAPK and not v_CHKREC)) or (v_ATR and not v_CHKREC))", 1.0)],
                                 [("(((v_ATM and not v_CHKREC) or (v_DNAPK and not v_CHKREC)) or (v_ATR and not v_CHKREC))", 1.0)],
                                 [("((((v_PCNATLS and not v_DSB) or (v_NHEJ and not v_DSB)) or (v_HRR and not v_DSB)) or not ((((((v_NHEJ or v_ICL) or v_PCNATLS) or v_HRR) or v_DSB) or v_CHKREC) or v_ADD))", 1.0)],
                                 [("((v_DSB and v_KU) and not v_CHKREC)", 1.0)],
                                 [("(((v_XPF and not (v_HRR or v_NHEJ)) or (v_DSB and not (v_HRR or v_NHEJ))) or (v_FAN1 and not (v_HRR or v_NHEJ)))", 1.0)],
                                 [("(v_MUS81 and v_FANCD2I)", 1.0)],
                                 [("((v_ssDNARPA and v_BRCA1) or ((v_FANCD2I and v_ssDNARPA) and not v_CHKREC))", 1.0)],
                                 [("((v_FAcore and ((v_ATM or v_ATR) or (v_DSB and v_H2AX))) and not v_USP1)", 1.0)],
                                 [("((v_ssDNARPA and (v_ATM or v_ATR)) or (v_ICL and (v_ATM or v_ATR)))", 1.0)],
                                 [("(v_ICL and not v_CHKREC)", 1.0)],
                                 [("((v_FANCM and (v_ATM or v_ATR)) and not v_CHKREC)", 1.0)],
                                 [("((v_DSB and ((v_ATM or v_DNAPK) or v_ATR)) and not v_CHKREC)", 1.0)],
                                 [("((v_DSB and ((v_BRCA1 and v_FANCD1N) and v_RAD51)) and not v_CHKREC)", 1.0)],
                                 [("(v_ICL and not v_DSB)", 1.0)],
                                 [("(v_DSB and not ((v_CHKREC or v_MRN) or v_FANCD2I))", 1.0)],
                                 [("((v_DSB and v_ATM) and not ((v_RAD51 or (v_KU and v_FANCD2I)) or v_CHKREC))", 1.0)],
                                 [("v_ICL", 1.0)],
                                 [("(((v_KU and (v_DNAPK and v_DSB)) and not (v_ATM and v_ATR)) or ((v_XPF and (v_DNAPK and v_DSB)) and not ((v_FANCJBRCA1 and v_ssDNARPA) or v_CHKREC)))", 1.0)],
                                 [("(((v_FAcore and v_ADD) and not (v_FAN1 or v_USP1)) or (v_ADD and not (v_FAN1 or v_USP1)))", 1.0)],
                                 [("((v_ssDNARPA and v_FANCD1N) and not v_CHKREC)", 1.0)],
                                 [("(((v_FANCD1N and v_FANCD2I) and not v_FANCM) or (v_PCNATLS and not v_FANCM))", 1.0)],
                                 [("(((v_p53 and v_MUS81) and not (v_FAcore and (v_FAN1 and v_FANCD2I))) or (v_MUS81 and not v_FANCM))", 1.0)],
                                 [("((((v_ATM and v_CHK2) and not v_CHKREC) or ((v_ATR and v_CHK1) and not v_CHKREC)) or (v_DNAPK and not v_CHKREC))", 1.0)],
                                 [("((v_DSB and ((v_FANCJBRCA1 and v_FANCD2I) or v_MRN)) and not (v_KU or v_RAD51))", 1.0)]])

print(type(env.env.env))
# set up logs
TOP_LEVEL_LOG_DIR = Path(args.log_dir)
TOP_LEVEL_LOG_DIR.mkdir(parents=True, exist_ok=True)

RUN_NAME = f"{args.env}_fa_brca_{args.exp_name}"

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
