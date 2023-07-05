import argparse
import random
from pathlib import Path

import gymnasium as gym
import gym_PBN
import numpy as np
import torch
from gym_PBN.utils.eval import compute_ssd_hist

import wandb
from ddqn_per import DDQNPER

model_cls = DDQNPER
model_name = "DDQNPER"

# Parse settings
parser = argparse.ArgumentParser(description="Train an RL model for target control.")
parser.add_argument(
    "--time-steps", metavar="N", type=int, help="Total number of training time steps."
)
parser.add_argument(
    "--learning-starts", type=int, metavar="LS", help="when the learning starts"
)
parser.add_argument(
    "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)."
)
parser.add_argument("--env", type=str, help="the environment to run.")
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
    "--exp-name", type=str, default="ddqn", metavar="E", help="the experiment name."
)
parser.add_argument("--log-dir", default="logs", help="path to save logs")
parser.add_argument(
    "--hyperparams", type=str, help="any extra hyper parameters for the model"
)
args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()
DEVICE = torch.device("cuda" if use_cuda else "cpu")
print(f"Training on {DEVICE}")

# Reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# Load env
#env = gym.make("gym-PBN/Bittner-28")
#this one was working

env = gym.make("gym-PBN/PBN-v0",
               logic_func_data=(
                   ["u", "x1", "x2", "x3", "x4"],
                   [
                       [],
                       [("x1", 1)],
                       [("x1 and not x1", 1)],  # perma False
                       [("x1 and not x1", 1)],  # perma False
                       [("x4", 1)],
                   ],
               ),
               goal_config={
                   "all_attractors": [{(0, 0, 0, 1)}, {(0, 0, 0, 0)}, {(1, 0, 0, 1)}, {(1, 0, 0, 0)}],
                   "target_nodes": {(0, 0, 0, 1)},
                   "intervene_on": ["x1", "x2", "x3", "x4"]
               },
               )

print("gym make")

# this one should be
# # https://github.com/sybila/biodivine-boolean-models/blob/main/models/%5Bid-095%5D__%5Bvar-9%5D__%5Bin-1%5D__%5BFISSION-YEAST-2008%5D/model.bnet
# env = gym.make("gym-PBN/PBCN-v0",
#                logic_func_data=(
#                    ['u', 'v_Cdc25', 'v_Cdc2_Cdc13', 'v_Cdc2_Cdc13_A', 'v_PP', 'v_Rum1', 'v_SK', 'v_Slp1', 'v_Ste9',
#                     'v_Wee1_Mik1', 'v_Start'],
#                    [
#                        [],
#                        [(
#                         " ((((not v_Cdc2_Cdc13  and  v_Cdc25)  and  not v_PP) or ((v_Cdc2_Cdc13  and  not v_Cdc25) and"
#                         "  not v_PP))  or  (v_Cdc2_Cdc13  and  v_Cdc25))",
#                         1)],
#                        [(" ((not v_Ste9  and  not v_Rum1)  and  not v_Slp1)", 1)],
#                        [(" ((((not v_Ste9  and  not v_Rum1)  and  not v_Slp1)  and  not v_Wee1_Mik1)  and  v_Cdc25)",
#                          1)],
#                        [(" v_Slp1", 1)],
#                        [(
#                         " ((((((((not v_SK  and  not v_Cdc2_Cdc13)  and  not v_Rum1)  and  "
#                         "not v_Cdc2_Cdc13_A)  and  v_PP)  or  (((not v_SK  and  not v_Cdc2_Cdc13)  and  v_Rum1)  and "
#                         " not v_Cdc2_Cdc13_A))  or  ((((not v_SK  and  not v_Cdc2_Cdc13)  and  v_Rum1)  and "
#                         " v_Cdc2_Cdc13_A)  and  v_PP))  or  ((((not v_SK  and  v_Cdc2_Cdc13)  and  v_Rum1)  and  "
#                         "not v_Cdc2_Cdc13_A)  and  v_PP))  or  ((((v_SK  and  not v_Cdc2_Cdc13)  and  v_Rum1)  and  "
#                         "not v_Cdc2_Cdc13_A)  and  v_PP))",
#                         1)],
#                        [(" v_Start", 1)],
#                        [(" v_Cdc2_Cdc13_A", 1)],
#                        [(
#                         " ((((((((not v_SK  and  not v_Cdc2_Cdc13)  and  not v_Ste9)  and  not v_Cdc2_Cdc13_A)  and "
#                         " v_PP)  or  (((not v_SK  and  not v_Cdc2_Cdc13)  and  v_Ste9)  and  not v_Cdc2_Cdc13_A))  or"
#                         "  ((((not v_SK  and  not v_Cdc2_Cdc13)  and  v_Ste9)  and  v_Cdc2_Cdc13_A)  and  v_PP))  or "
#                         " ((((not v_SK  and  v_Cdc2_Cdc13)  and  v_Ste9)  and  not v_Cdc2_Cdc13_A)  and  v_PP))  or  "
#                         "((((v_SK  and  not v_Cdc2_Cdc13)  and  v_Ste9)  and  not v_Cdc2_Cdc13_A)  and  v_PP))",
#                         1)],
#                        [(
#                         " ((((not v_Cdc2_Cdc13  and  not v_Wee1_Mik1)  and  v_PP)  or  (not v_Cdc2_Cdc13  and "
#                         " v_Wee1_Mik1))  or  ((v_Cdc2_Cdc13  and  v_Wee1_Mik1)  and  v_PP))",
#                         1)],
#                        [('v_Start', 1)],
#                    ],
#                ),
#                goal_config={
#                    "all_attractors": [],
#                    "target_nodes": {(0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0)},
#                    "intervene_on": []
#                },
#                )
print("gym made")

print("where is the graph?")
# set up logs
TOP_LEVEL_LOG_DIR = Path(args.log_dir)
TOP_LEVEL_LOG_DIR.mkdir(parents=True, exist_ok=True)

RUN_NAME = f"{args.env.split('/')[-1]}_{args.exp_name}_{args.seed}"

# Checkpoints
checkpoint_path = Path(args.checkpoint_dir) / RUN_NAME
checkpoint_path.mkdir(parents=True, exist_ok=True)


def get_latest_checkpoint():
    files = list(checkpoint_path.glob("*.pt"))
    if len(files) > 0:
        return max(files, key=lambda x: x.stat().st_ctime)
    else:
        return None


# Model
total_time_steps = args.time_steps
resume_steps = 0
hyperparams = {}
if args.hyperparams:
    hyperparams = {
        param.split("=")[0]: eval(param.split("=")[1])
        for param in args.hyperparams.split(",")
    }
model = DDQNPER(env, DEVICE, **hyperparams)

config = model.get_config()
config["learning_starts"] = args.learning_starts
run = wandb.init(
    project="pbn-rl",
#    entity="uos-plccn",
    sync_tensorboard=True,
    monitor_gym=True,
    config=config,
    name=RUN_NAME,
    save_code=True,
)

if args.resume_training:
    model_path = get_latest_checkpoint()

    if model_path:
        print(f"Loading model {model_path}.")
        model = model_cls.load(model_path, env, device=DEVICE)
        resume_steps = total_time_steps - model.num_timesteps


if not args.eval_only:
    print(f"Training for {total_time_steps - resume_steps} time steps...")
    model.learn(
        total_time_steps,
        learning_starts=args.learning_starts,
        checkpoint_freq=500,
        checkpoint_path=checkpoint_path,
        resume_steps=resume_steps,
        log_dir=TOP_LEVEL_LOG_DIR,
        log_name=RUN_NAME,
        log=True,
        run=run,
    )

print(f"Evaluating...")
ssd, plot = compute_ssd_hist(env, model, resets=300, iters=100_000, multiprocess=False)
run.log({"SSD": plot})

from itertools import product

count = 0
for state in product([0, 1], repeat=5):
    print(f"for state={state} we got {model.predict(state, deterministic=True, show_work=False)}")
    count += 1
    if count > 10:
        break


env.close()
run.finish()
