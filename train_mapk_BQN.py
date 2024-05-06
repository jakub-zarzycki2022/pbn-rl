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
# DRUG-SYNERGY-PREDICTION
# from https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2020.00862/full
env = gym.make(f"gym-PBN/PBNEnv",
               N=args.size,
               genes=[
                   

               ],
               logic_functions=[
                    [("v_BMPR2", 1.0)],
                    [(" not v_AKT_f", 1.0)],
                    [("((v_ILK  or  (v_PDPK1  or  v_mTORC2_c))  and   not v_PPP1CA)", 1.0)],
                    [("(v_SMAD3  or  (v_JUN  or  (v_SMAD4  or  (v_ATF2  or  v_FOS))))", 1.0)],
                    [("v_ROCK1", 1.0)],
                    [("(v_ERK_f  or  (v_MAPK14  or  v_JNK_f))", 1.0)],
                    [("(v_GSK3_f  and   not (v_LRP_f  or  (v_PPM1A  or  v_PPP1CA)))", 1.0)],
                    [("(v_ISGF3_c  or  (v_FOXO_f  or  v_CASP3))", 1.0)],
                    [(" not (v_AKT_f  or  v_RSK_f)", 1.0)],
                    [("v_TP53", 1.0)],
                    [(" not v_BAD", 1.0)],
                    [(" not (v_SMURF1  or  v_SMURF2)", 1.0)],
                    [("((v_GSK3_f  or  (v_CK1_f  or  v_AXIN1))  and   not v_LRP_f)", 1.0)],
                    [("(v_CASP8  or  v_CASP9)", 1.0)],
                    [(" not v_CFLAR", 1.0)],
                    [("(v_CYCS  or  v_PPP1CA)", 1.0)],
                    [("((v_CREBBP  or  v_EP300)  and   not v_TP53)", 1.0)],
                    [("(v_STAT3  or  (v_TCF7_f  or  v_RSK_f))", 1.0)],
                    [("(v_SRC  and   not v_ARHGAP24)", 1.0)],
                    [("(v_AKT_f  and   not v_ITCH)", 1.0)],
                    [("(v_LIMK1  or  v_LIMK2)", 1.0)],
                    [("(v_PRKCA  or  (v_AKT_f  or  v_TRAF6))", 1.0)],
                    [(" not v_LRP_f", 1.0)],
                    [("v_CHUK", 1.0)],
                    [("v_PRKACA", 1.0)],
                    [("(v_CHUK  and   not v_BTRC)", 1.0)],
                    [("(v_BAX  and   not v_BCL2)", 1.0)],
                    [("v_DVL_f", 1.0)],
                    [("v_DKK_g", 1.0)],
                    [("(v_TCF7_f  and   not v_MYC)", 1.0)],
                    [("((v_DUSP1_g  or  (v_MAPK14  or  v_MSK_f))  and   not v_SKP2)", 1.0)],
                    [("(v_ERK_f  or  v_MAPK14)", 1.0)],
                    [("(v_ERK_f  or  v_mTORC1_c)", 1.0)],
                    [("((v_FZD_f  or  v_SMAD1)  and   not v_ITCH)", 1.0)],
                    [(" not v_TCF7_f", 1.0)],
                    [("(v_AKT_f  and   not (v_PRKCD  or  v_SKI))", 1.0)],
                    [("(v_MEK_f  and   not (v_DUSP6  or  v_PPP1CA))", 1.0)],
                    [("(v_ERK_f  or  (v_SRF  or  v_RSK_f))", 1.0)],
                    [(" not (v_NLK  or  (v_CK1_f  or  v_AKT_f))", 1.0)],
                    [(" not v_SFRP1", 1.0)],
                    [("(v_GRB2  and   not v_ERK_f)", 1.0)],
                    [(" not v_MAPK14", 1.0)],
                    [("v_SHC1", 1.0)],
                    [(" not (v_ERK_f  or  (v_MAPK14  or  (v_AKT_f  or  (v_RSK_f  or  (v_S6K_f  or  v_DVL_f)))))", 1.0)],
                    [("(v_MAP3K7  and   not (v_PLK1  or  (v_PPM1A  or  v_TP53)))", 1.0)],
                    [("v_PAK1", 1.0)],
                    [("(v_LIF  or  v_AP1_c)", 1.0)],
                    [("(v_ILR_f  and   not v_SOCS1)", 1.0)],
                    [(" not (v_ERK_f  or  (v_S6K_f  or  v_IKBKB))", 1.0)],
                    [("(v_STAT2  or  v_STAT1)", 1.0)],
                    [("v_JNK_f", 1.0)],
                    [("(v_ILR_f  and   not (v_SOCS1  or  v_PTPN6))", 1.0)],
                    [("((v_MAP2K7  or  (v_MAP2K4  or  v_PAK1))  and   not v_DUSP1)", 1.0)],
                    [("(v_JNK_f  and   not v_GSK3_f)", 1.0)],
                    [("(v_PTPN11  or  v_SOS1)", 1.0)],
                    [("v_CTNNB1", 1.0)],
                    [("v_RAF_f", 1.0)],
                    [("(v_ROCK1  or  v_RAC_f)", 1.0)],
                    [("(v_ROCK1  and   not v_PRKCD)", 1.0)],
                    [("((v_ERK_f  or  (v_MAPK14  or  (v_JNK_f  or  v_FZD_f)))  and   not v_DKK_f)", 1.0)],
                    [("(v_MAP3K7  or  v_MAP3K5)", 1.0)],
                    [("(v_MAP3K7  or  (v_MAP3K11  or  (v_MAP3K4  or  v_GRAP2)))", 1.0)],
                    [("(v_MAP3K7  or  (v_MAPK8IP3  or  v_GRAP2))", 1.0)],
                    [("v_RAC_f", 1.0)],
                    [("v_RAC_f", 1.0)],
                    [(" not v_AKT_f", 1.0)],
                    [("v_TAB_f", 1.0)],
                    [("v_IKBKB", 1.0)],
                    [("((v_MAP2K3  or  v_MAP2K4)  and   not v_DUSP1)", 1.0)],
                    [("v_ROCK1", 1.0)],
                    [("v_MAPK14", 1.0)],
                    [("((v_MAPKAPK2  or  (v_AKT_f  or  (v_MDM2_g  or  v_PPP1CA)))  and   not v_S6K_f)", 1.0)],
                    [("(v_NFKB_f  or  v_TP53)", 1.0)],
                    [("((v_RAF_f  or  v_MAP3K8)  and   not v_ERK_f)", 1.0)],
                    [("(v_STAT3  or  v_LEF1)", 1.0)],
                    [("(v_ERK_f  or  v_MAPK14)", 1.0)],
                    [("((v_STAT3  or  (v_TCF7_f  or  v_PLK1))  and   not v_GSK3_f)", 1.0)],
                    [("(v_REL_f  or  (v_MSK_f  or  (v_CHUK  or  v_IKBKB)))", 1.0)],
                    [("v_MAP3K7", 1.0)],
                    [("(v_RAC_f  or  v_CDC42)", 1.0)],
                    [("(v_TGFBR2  or  v_TGFBR1)", 1.0)],
                    [("(v_PIK3CA  and   not v_PTEN)", 1.0)],
                    [("v_MAPKAPK2", 1.0)],
                    [("(v_KRAS  or  (v_GAB_f  or  v_IRS1))", 1.0)],
                    [("v_SYK", 1.0)],
                    [("(v_MAPKAPK2  or  v_PDPK1)", 1.0)],
                    [("v_PTEN", 1.0)],
                    [("(v_SMAD7  and   not v_RTPK_f)", 1.0)],
                    [("(v_NFKB_f  or  v_FOS)", 1.0)],
                    [("v_PLCG1", 1.0)],
                    [("(v_PDPK1  or  v_CASP3)", 1.0)],
                    [("((v_PTEN_g  or  v_ROCK1)  and   not (v_SRC  or  (v_GSK3_f  or  v_CBPp300_c)))", 1.0)],
                    [("v_EGR1", 1.0)],
                    [("v_GAB_f", 1.0)],
                    [("v_SRC", 1.0)],
                    [("(v_CCND1  or  (v_MYC  or  v_RSK_f))", 1.0)],
                    [("((v_VAV1  or  (v_TIAM1  or  (v_mTORC2_c  or  v_DVL_f)))  and   not v_ARHGAP24)", 1.0)],
                    [("(v_KRAS  and   not (v_ERK_f  or  (v_RHEB  or  v_AKT_f)))", 1.0)],
                    [("((v_CBPp300_c  or  (v_MSK_f  or  v_IKBKB))  and   not v_STAT1)", 1.0)],
                    [(" not v_TSC_f", 1.0)],
                    [("(v_DAAM1  and   not (v_SMURF1  or  (v_PARD6A  or  (v_RAC_f  or  v_RND3))))", 1.0)],
                    [("v_ROCK1", 1.0)],
                    [("(v_RHOA  or  v_CASP3)", 1.0)],
                    [("(v_ERK_f  or  v_PDPK1)", 1.0)],
                    [("((v_RTPK_g  or  v_MMP_f)  and   not (v_MAPK14  or  v_MEK_f))", 1.0)],
                    [("v_FOXO_f", 1.0)],
                    [("(v_mTORC1_c  or  v_PDPK1)", 1.0)],
                    [("v_SFRP1_g", 1.0)],
                    [(" not v_MYC", 1.0)],
                    [("((v_RTPK_f  or  (v_SRC  or  (v_ILR_f  or  v_TGFBR1)))  and   not v_PTEN)", 1.0)],
                    [(" not v_AKT_f", 1.0)],
                    [("(v_CCND1  or  (v_ERK_f  or  v_EP300))", 1.0)],
                    [("(v_ACVR1  and   not (v_SMAD6  or  (v_ERK_f  or  (v_SMURF1  or  (v_GSK3_f  or  (v_PPM1A  or  v_SKI))))))", 1.0)],
                    [("((v_ITCH  or  (v_ERK_f  or  (v_CBPp300_c  or  (v_TGFBR1  or  v_ACVR1))))  and   not (v_PPM1A  or  (v_SMURF2  or  v_SKI)))", 1.0)],
                    [("((v_MAPK14  or  (v_JNK_f  or  (v_TGFBR1  or  v_ACVR1)))  and   not (v_SMAD6  or  (v_ERK_f  or  (v_GSK3_f  or  (v_AKT_f  or  (v_SMAD7  or  (v_PPM1A  or  v_SKI)))))))", 1.0)],
                    [("((v_SMAD3  or  (v_ERK_f  or  (v_PIAS1  or  (v_SMAD5  or  (v_SMAD2  or  v_SMAD1)))))  and   not (v_SMAD6  or  (v_SMURF1  or  (v_SMAD7  or  v_SKI))))", 1.0)],
                    [("(v_ACVR1  and   not (v_SMURF2  or  v_SKI))", 1.0)],
                    [("v_SMAD6_g", 1.0)],
                    [("(v_SMAD3  or  (v_SMAD4  or  v_SMAD2))", 1.0)],
                    [("((v_SMAD7_g  or  (v_SMURF1  or  v_EP300))  and   not (v_ITCH  or  (v_AXIN1  or  v_SMURF2)))", 1.0)],
                    [("(v_SMAD3  or  (v_SMAD4  or  v_SMAD2))", 1.0)],
                    [("v_SMAD7", 1.0)],
                    [("v_SMAD7", 1.0)],
                    [("v_SOCS1_g", 1.0)],
                    [("v_STAT1", 1.0)],
                    [("((v_PLCG1  or  v_GRB2)  and   not v_ERK_f)", 1.0)],
                    [("(v_RTPK_f  and   not v_CSK)", 1.0)],
                    [("(v_MAPKAPK2  or  (v_CFL_f  or  v_RSK_f))", 1.0)],
                    [("((v_PRKCD  or  (v_SRC  or  (v_JAK_f  or  (v_MAPK14  or  v_IKBKB))))  and   not v_PIAS1)", 1.0)],
                    [("v_JAK_f", 1.0)],
                    [("((v_PRKCD  or  (v_SRC  or  (v_JAK_f  or  (v_ERK_f  or  (v_MAPK14  or  (v_JNK_f  or  (v_mTORC1_c  or  v_IRAK1)))))))  and   not v_PPP1CA)", 1.0)],
                    [("v_ILR_f", 1.0)],
                    [("(v_TRAF6  and   not v_MAPK14)", 1.0)],
                    [("(v_CTNNB1  and   not v_NLK)", 1.0)],
                    [("(v_JUN  or  (v_NFKB_f  or  v_FOS))", 1.0)],
                    [("(v_TGFBR2  and   not (v_SMAD6  or  (v_SMURF1  or  (v_SMAD7  or  v_SMURF2))))", 1.0)],
                    [("(v_TGFB1  and   not (v_SMURF1  or  v_SMURF2))", 1.0)],
                    [(" not v_ROCK1", 1.0)],
                    [("((v_PRKCD  or  (v_MAPK14  or  (v_PIAS1  or  v_EP300)))  and   not v_MDM2)", 1.0)],
                    [("(v_TGFBR1  or  v_IRAK1)", 1.0)],
                    [("(v_GSK3_f  and   not (v_ERK_f  or  (v_AKT_f  or  (v_RSK_f  or  v_IKBKB))))", 1.0)],
                    [("v_SYK", 1.0)],
                    [("((v_RHEB  or  v_RSK_f)  and   not v_AKT1S1)", 1.0)],
                    [("((v_PIK3CA  or  v_TSC_f)  and   not v_S6K_f)", 1.0)],
               ])

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
