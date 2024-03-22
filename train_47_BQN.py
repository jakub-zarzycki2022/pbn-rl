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
# A systems pharmacology model for inflammatory bowel disease
# https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0192949
env = gym.make(f"gym-PBN/PBNEnv",
               N=args.size,
               genes=['v_CD4_NKG2D', 'v_CD4_NKG2Dupregulation', 'v_CD8_NKG2D', 'v_DC', 'v_DEF', 'v_FIBROBLAST', 
                      'v_GRANZB', 'v_IEC_MICA_B', 'v_IEC_MICA_Bupregulation', 'v_IEC_ULPB1_6', 'v_IFNg', 'v_IL10', 
                      'v_IL12', 'v_IL13', 'v_IL15', 'v_IL17', 'v_IL18', 'v_IL1b', 'v_IL2', 'v_IL21', 'v_IL22', 
                      'v_IL22upregulation', 'v_IL23', 'v_IL4', 'v_IL6', 'v_LPS', 'v_MACR', 'v_MDP', 'v_MMPs', 'v_NFkB', 
                      'v_NK', 'v_NK_NKG2D', 'v_NOD2', 'v_PERFOR', 'v_PGN', 'v_TGFb', 'v_TLR2', 'v_TLR4', 'v_TNFa', 
                      'v_Th0', 'v_Th0_M', 'v_Th1', 'v_Th17', 'v_Th17_M', 'v_Th2', 'v_Th2upregulation', 'v_Treg'],
               logic_functions=[[('((v_PGN and not (v_CD4_NKG2D and ((v_IEC_ULPB1_6 or v_IEC_MICA_B) or v_IL10))) or ((v_MDP and not (v_CD4_NKG2D and ((v_IEC_ULPB1_6 or v_IEC_MICA_B) or v_IL10))) or (((v_CD4_NKG2D and ((v_TNFa or v_IL15) and not v_CD4_NKG2Dupregulation)) and not (v_CD4_NKG2D and ((v_IEC_ULPB1_6 or v_IEC_MICA_B) or v_IL10))) or (v_LPS and not (v_CD4_NKG2D and ((v_IEC_ULPB1_6 or v_IEC_MICA_B) or v_IL10))))))', 1.0)],
                               [('(v_CD4_NKG2D and (v_TNFa or v_IL15))', 1.0)],
                               [('((v_MDP and not (v_CD8_NKG2D and (v_IEC_MICA_B or (v_IEC_ULPB1_6 or (v_IL21 and v_IL2))))) or ((v_PGN and not (v_CD8_NKG2D and (v_IEC_MICA_B or (v_IEC_ULPB1_6 or (v_IL21 and v_IL2))))) or (v_LPS and not (v_CD8_NKG2D and (v_IEC_MICA_B or (v_IEC_ULPB1_6 or (v_IL21 and v_IL2)))))))', 1.0)],
                               [('((v_TLR2 and not (v_DC and v_IL10)) or ((v_TLR4 and not (v_DC and v_IL10)) or (v_NOD2 and not (v_DC and v_IL10))))', 1.0)],
                               [('(v_IL22 or (v_IL17 or v_NOD2))', 1.0)],
                               [('((v_IL2 and not (v_FIBROBLAST and (v_IL12 or v_IFNg))) or ((v_MACR and (v_TGFb or (v_IL13 or v_IL4))) and not (v_FIBROBLAST and (v_IL12 or v_IFNg))))', 1.0)],
                               [('(v_NK_NKG2D or (v_CD8_NKG2D or (v_NK or (v_DC and (not v_PGN or not v_LPS)))))', 1.0)],
                               [('(((v_IEC_MICA_B and (v_TNFa and not v_IEC_MICA_Bupregulation)) and not v_TGFb) or ((v_LPS and not v_TGFb) or ((v_PGN and not v_TGFb) or (v_MDP and not v_TGFb))))', 1.0)],
                               [('(v_IEC_MICA_B and v_TNFa)', 1.0)],
                               [('(v_CD8_NKG2D and (v_PGN or (v_LPS or v_MDP)))', 1.0)],
                               [('((((v_Th17 and (v_PGN or (v_LPS or v_MDP))) and not (v_IFNg and (v_TGFb or v_IL10))) and not v_Th2) or (((v_Th1 and not (v_IFNg and (v_TGFb or v_IL10))) and not v_Th2) or ((((v_IL18 and (v_IL12 and (v_Th0 or v_MACR))) and not (v_IFNg and (v_TGFb or v_IL10))) and not v_Th2) or ((((v_IL23 and ((v_PGN or (v_LPS or v_MDP)) and v_NK)) and not (v_IFNg and (v_TGFb or v_IL10))) and not v_Th2) or ((((v_NK_NKG2D and (v_IEC_ULPB1_6 or v_IEC_MICA_B)) and not (v_IFNg and (v_TGFb or v_IL10))) and not v_Th2) or (((v_CD8_NKG2D and (v_IEC_ULPB1_6 or v_IEC_MICA_B)) and not (v_IFNg and (v_TGFb or v_IL10))) and not v_Th2))))))', 1.0)],
                               [('(v_Treg or ((v_DC and v_LPS) or ((v_TLR2 and (v_NFkB and (not v_MACR and not v_IFNg))) or ((v_MACR and (v_LPS and not v_IL4)) or (v_Th2 and not v_IL23)))))', 1.0)],
                               [('((v_LPS and (v_IFNg and ((v_MACR and v_PGN) or v_DC))) or (v_TLR2 and (v_NFkB and (v_DC or v_MACR))))', 1.0)],
                               [('v_Th2', 1.0)],
                               [('((v_FIBROBLAST and (v_PGN or (v_LPS or v_MDP))) or (v_MACR and (v_IFNg or v_LPS)))', 1.0)],
                               [('(((v_Th17_M and (v_PGN or (v_LPS or v_MDP))) and not (v_IL17 and (v_TGFb or v_IL13))) or (((v_CD4_NKG2D and (v_IEC_ULPB1_6 or v_IEC_MICA_B)) and not (v_IL17 and (v_TGFb or v_IL13))) or (v_Th17 and not (v_IL17 and (v_TGFb or v_IL13)))))', 1.0)],
                               [('(v_LPS and (v_NFkB and (v_DC or v_MACR)))', 1.0)],
                               [('(((v_MACR and (v_NFkB and v_LPS)) and not (v_IL10 and v_IL1b)) or ((v_DC and (v_NFkB and v_LPS)) and not (v_IL10 and v_IL1b)))', 1.0)],
                               [('((v_Th0_M and (v_PGN or (v_LPS or v_MDP))) or (v_Th0 or v_DC))', 1.0)],
                               [('(((((v_Th0 and v_IL6) and not v_IFNg) and not v_IL4) and not v_TGFb) or v_Th17)', 1.0)],
                               [('(v_Th17 or ((v_NK and v_IL23) or (((v_Th0 and (v_IL22 and (v_IL21 and not v_IL22upregulation))) and not v_TGFb) or (v_CD4_NKG2D or (v_NK and (v_IL18 and v_IL12))))))', 1.0)],
                               [('(v_Th0 and (v_IL21 and v_IL22))', 1.0)],
                               [('((v_MACR and v_IL1b) or v_DC)', 1.0)],
                               [('v_Th2', 1.0)],
                               [('((v_DC and (v_PGN or v_LPS)) or ((v_MACR and v_PGN) or ((v_Th17 and v_IL23) or (v_NFkB and (not v_IL10 or not v_IL4)))))', 1.0)],
                               [('not (v_GRANZB or (v_DEF or v_PERFOR))', 1.0)],
                               [('((v_NOD2 and not (v_MACR and v_IL10)) or ((v_IFNg and not (v_MACR and v_IL10)) or ((v_IL15 and not (v_MACR and v_IL10)) or ((v_TLR4 and not (v_MACR and v_IL10)) or (v_TLR2 and not (v_MACR and v_IL10))))))', 1.0)],
                               [('not (v_PERFOR or (v_DEF or v_GRANZB))', 1.0)],
                               [('((v_FIBROBLAST and (v_TNFa or (v_IL21 or (v_IL17 or v_IL1b)))) or (v_MACR and v_TNFa))', 1.0)],
                               [('(v_NOD2 or (v_TLR4 or v_TLR2))', 1.0)],
                               [('(((v_DC and v_IL15) and not (v_NK and v_Treg)) or ((v_IL23 and not (v_NK and v_Treg)) or ((v_IL18 and v_IL10) and not (v_NK and v_Treg))))', 1.0)],
                               [('((v_MDP and not (v_NK_NKG2D and (v_IEC_ULPB1_6 and (v_TGFb and (v_IEC_MICA_B and (v_IL21 and v_IL12)))))) or ((v_PGN and not (v_NK_NKG2D and (v_IEC_ULPB1_6 and (v_TGFb and (v_IEC_MICA_B and (v_IL21 and v_IL12)))))) or (v_LPS and not (v_NK_NKG2D and (v_IEC_ULPB1_6 and (v_TGFb and (v_IEC_MICA_B and (v_IL21 and v_IL12))))))))', 1.0)],
                               [('v_MDP', 1.0)],
                               [('(v_NK_NKG2D or v_NK)', 1.0)],
                               [('not (v_GRANZB or (v_DEF or v_PERFOR))', 1.0)],
                               [('(v_Treg or v_MACR)', 1.0)],
                               [('v_PGN', 1.0)],
                               [('v_LPS', 1.0)],
                               [('(((v_MACR and (v_IL2 or (v_PGN or (v_IFNg and v_LPS)))) and not (v_IL10 and (v_TNFa and (v_TLR4 or v_TLR2)))) or (((v_CD8_NKG2D and (v_IEC_ULPB1_6 or v_IEC_MICA_B)) and not (v_IL10 and (v_TNFa and (v_TLR4 or v_TLR2)))) or (((v_FIBROBLAST and v_IFNg) and not (v_IL10 and (v_TNFa and (v_TLR4 or v_TLR2)))) or (((v_NFkB and v_LPS) and not (v_IL10 and (v_TNFa and (v_TLR4 or v_TLR2)))) or (((v_CD4_NKG2D and (v_IEC_ULPB1_6 or v_IEC_MICA_B)) and not (v_IL10 and (v_TNFa and (v_TLR4 or v_TLR2)))) or (((v_NK and ((v_PGN or (v_LPS or v_MDP)) and (v_IL23 or (v_IL15 or v_IL2)))) and not (v_IL10 and (v_TNFa and (v_TLR4 or v_TLR2)))) or ((v_NK_NKG2D and (v_IEC_ULPB1_6 or v_IEC_MICA_B)) and not (v_IL10 and (v_TNFa and (v_TLR4 or v_TLR2))))))))))', 1.0)],
                               [('(v_MDP or (v_PGN or v_LPS))', 1.0)],
                               [('(v_Th0_M or (v_Th0 and (v_IL23 or v_IL12)))', 1.0)],
                               [('((v_Th0 and (v_IL18 or (v_IL12 or v_IFNg))) and not (v_Th1 and (v_IL4 or (v_TGFb or (v_IL10 or (v_Treg or (v_Th2 or (v_IL12 and (v_IL23 or v_IL17)))))))))', 1.0)],
                               [('((v_Th0 and (v_IL23 or (v_IL6 or v_IL1b))) and not (v_Th17 and (v_IL12 or (v_TGFb or (v_Treg or (v_IFNg or v_IL4))))))', 1.0)],
                               [('((v_Th0_M and ((v_PGN or (v_LPS or v_MDP)) and ((v_IL6 and v_IL1b) or (v_IL23 or v_IL2)))) or v_Th17_M)', 1.0)],
                               [('((v_Th0 and (((v_Th2 and v_IL4) and not v_Th2upregulation) or (((v_IL18 and v_IL4) and not v_IL12) or v_IL10))) and not (v_Th2 and (v_TGFb or (v_Treg or v_IFNg))))', 1.0)],
                               [('(v_Th2 and v_IL4)', 1.0)],
                               [('((v_Th0 and (v_TGFb or v_TLR2)) and not (v_Treg and (v_IL22 or (v_IL23 or (v_TNFa or (v_IL21 or (v_IL6 or v_Th17)))))))', 1.0)]])

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
model = model_cls((state_len, state_len), state_len + 1, config, env)
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
