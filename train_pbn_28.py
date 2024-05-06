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
from ddqn_per import DDQNPER, DDQN

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

# # Load env
# env_bittner = gym.make("gym-PBN/Bittner-28")

# #this one was working
# env_simple = gym.make("gym-PBN/PBN-v0",
#                logic_func_data=(
#                    ["u", "x1", "x2", "x3", "x4"],
#                    [
#                        [],
#                        [("x1", 1)],
#                        [("False", 1)],  # perma False
#                        [("False", 1)],  # perma False
#                        [("x4", 1)],
#                    ],
#                ),
#                goal_config={
#                    "all_attractors": [],
#                    "target_nodes": {(0, 0, 0, 0, 1)},
#                    "intervene_on": ["x1", "x2", "x3", "x4"]
#                },
#                )

# print("gym make")

# this one should be
# https://github.com/sybila/biodivine-boolean-models/blob/main/models/%5Bid-095%5D__%5Bvar-9%5D__%5Bin-1%5D__%5BFISSION-YEAST-2008%5D/model.bnet
# env_yeast = gym.make(
#                "gym-PBN/PBN-v0",
#                name="yeast",
#                logic_func_data=(
#                    ['u',
#                     'v_Cdc25', 'v_Cdc2_Cdc13', 'v_Cdc2_Cdc13_A', 'v_PP', 'v_Rum1',
#                     'v_SK', 'v_Slp1', 'v_Ste9', 'v_Wee1_Mik1', 'v_Start'],
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
#print("gym made")

# env bittner-7 does not work. I should really look into it
# TODO: look into it
# env_melanoma_7 = gym.make("gym-PBN/Bittner-7-v0")

# env_manual_melanoma = gym.make("gym-PBN/PBN-v0",
#                logic_func_data=(
#                    ["u", 'x234237', 'x324901', 'x759948', 'x25485', 'x266361', 'x108208', 'x130057'],
#                    [
#                         [],
#                         [('not x108208 and  not x130057 and  not x266361', 0.2), ('not x108208 and  not x130057 and  not x266361', 0.2), ('not x108208 and  not x130057 and  not x266361', 0.2), ('not x108208 and  not x130057 and  not x266361', 0.2), ('not x108208 and  not x130057 and  not x266361', 0.2)],
#                         [('x324901 and  not x108208 and  not x130057 and  not x266361', 0.2), ('x324901 and  not x108208 and  not x130057 and  not x266361', 0.2), ('x324901 and  not x108208 and  not x130057 and  not x266361', 0.2), ('( x324901 and  not x108208 and  not x130057 and  not x266361) or  ( not x108208 and  not x130057 and  not x234237 and  not x266361)', 0.2), ('( x324901 and  not x108208 and  not x130057 and  not x266361) or  ( not x108208 and  not x130057 and  not x234237 and  not x266361)', 0.2)],
#                         [('not x108208 and  not x130057 and  not x266361', 0.2), ('not x108208 and  not x130057 and  not x266361', 0.2), ('not x108208 and  not x130057 and  not x266361', 0.2), ('not x108208 and  not x130057 and  not x266361', 0.2), ('not x108208 and  not x130057 and  not x266361', 0.2)],
#                         [('( x25485 and  not x108208 and  not x130057 and  not x266361) or  ( x234237 and  not x108208 and  not x130057 and  not x266361 and  not x759948)', 0.2), ('( x234237 and  not x108208 and  not x130057 and  not x266361) or  ( x25485 and  not x108208 and  not x130057 and  not x266361) or  ( not x108208 and  not x130057 and  not x266361 and  not x759948)', 0.2), ('( x25485 and  not x108208 and  not x130057 and  not x266361) or  ( not x108208 and  not x130057 and  not x266361 and  not x324901 and  not x759948)', 0.2), ('( x25485 and  not x108208 and  not x130057 and  not x266361) or  ( x234237 and  not x108208 and  not x130057 and  not x266361 and  not x324901 and  not x759948)', 0.2), ('( x25485 and  not x108208 and  not x130057 and  not x266361) or  ( x234237 and  not x108208 and  not x130057 and  not x266361 and  not x759948)', 0.2)],
#                         [('not x108208 and  not x130057 and  not x266361', 0.2), ('not x108208 and  not x130057 and  not x266361', 0.2), ('not x108208 and  not x130057 and  not x266361', 0.2), ('not x108208 and  not x130057 and  not x266361', 0.2), ('not x108208 and  not x130057 and  not x266361', 0.2)],
#                         [('not x108208 and  not x130057 and  not x266361', 0.2), ('not x108208 and  not x130057 and  not x266361', 0.2), ('not x108208 and  not x130057 and  not x266361', 0.2), ('not x108208 and  not x130057 and  not x266361', 0.2), ('not x108208 and  not x130057 and  not x266361', 0.2)],
#                         [('not x108208 and  not x130057 and  not x266361', 0.2), ('not x108208 and  not x130057 and  not x266361', 0.2), ('not x108208 and  not x130057 and  not x266361', 0.2), ('not x108208 and  not x130057 and  not x266361', 0.2), ('not x108208 and  not x130057 and  not x266361', 0.2)],
#                    ],
#                ),
#                goal_config={
#                    "all_attractors": [],
#                    "target_nodes": {(0, 1, 0, 0, 0, 0, 0, 0)},
#                    "intervene_on": ['x234237', 'x234237', 'x759948', 'x25485', 'x266361', 'x108208', 'x130057']
#                },
#                )
#

env_pbn10 = gym.make("gym-PBN/Bittner-28")

env = env_pbn10

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


def state_equals(state1, state2):
    for i in range(len(state2)):
        if state1[i] != state2[i]:
            return False
    return True


# Model
total_time_steps = args.time_steps
print(total_time_steps)
resume_steps = 0
hyperparams = {}
if args.hyperparams:
    print(f"got {hyperparams}")
    hyperparams = {
        param.split("=")[0]: eval(param.split("=")[1])
        for param in args.hyperparams.split(",")
    }

print(hyperparams)
hyperparams["policy_kwargs"] = {"net_arch": [(50, 50)]}
hyperparams["buffer_size"]: int = 64
hyperparams["batch_size"]: int = 64
hyperparams["target_update"]: int = 512
hyperparams["gamma"] = 0.9
hyperparams["max_epsilon"]: float = 1.0
hyperparams["min_epsilon"]: float = 0.05
hyperparams["exploration_fraction"]: float = 0.1
hyperparams["learning_rate"]: float = 0.0001
hyperparams["max_grad_norm"]: float = 10.0

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
    print("worked")
    model_path = get_latest_checkpoint()

    if model_path:
        print(f"Loading model {model_path}.")
        model = model_cls.load(model_path, env, device=DEVICE)
        resume_steps = total_time_steps - model.num_timesteps
        if model is None:
            raise ValueError


if not args.eval_only:
    print(f"Training for {total_time_steps - resume_steps} time steps...")
    model.learn(
        total_time_steps,
        learning_starts=args.learning_starts,
        checkpoint_freq=1000,
        checkpoint_path=checkpoint_path,
        resume_steps=resume_steps,
        log_dir=TOP_LEVEL_LOG_DIR,
        log_name=RUN_NAME,
        log=True,
        run=run,
    )

print(f"Evaluating...")
ssd, plot = compute_ssd_hist(env, model, resets=300, iters=100_000, multiprocess=True)
run.log({"SSD": plot})

attrs = env.all_attractors
target = env.target_nodes
lens = []
actions = defaultdict(list)

print("testig the model")
print(f"target is {target}")
for attractor in env.env.env.env.all_attractors:
    for initial_state in attractor:
        state = initial_state
        state = [0 if i == '*' else i for i in list(state)]
        _ = env.reset()
        _ = env.env.env.env.graph.setState(state)
        count = 0
        while not env.in_target(state):
            count += 1
            action = model.predict(state)
            actions[initial_state].append(action)
            _ = env.step(action)
            state = env.render()
        print(f"for initial state {initial_state} got {actions[initial_state]} (total of {count} steps)")
        lens.append(count)


env.close()
run.finish()
