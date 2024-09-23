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
from sympy import symbols
from sympy.logic import SOPform

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


def translate(sym, logic_function):
    """
    We need variable names to start with letter.

    """
    if logic_function == 'True':
        return f'{sym[0]} or not {sym[0]}'

    if logic_function == 'False':
        return f'{sym[0]} and not {sym[0]}'

    logic_function = logic_function.replace('~', "not ")
    logic_function = logic_function.replace('|', " or ")
    logic_function = logic_function.replace('&', " and ")

    return logic_function


with open(args.assa_file, "r") as env_file:

    genes = []

    line_no = 0
    # skip two headear lines
    _ = next(env_file)
    _ = next(env_file)

    line_no += 2

    line = next(env_file)
    line_no += 1
    n_genes = int(line)

    line = next(env_file)
    line_no += 1
    number_of_functions = line.split()
    number_of_functions = [int(i) for i in number_of_functions]

    line = next(env_file)
    line_no += 1

    n_predictors = ([int(i) for i in (line.split())])

    truth_tables = defaultdict(list)
    fun_id = 0

    # get truth tables of logic functions
    # some genes have more than one function - we deal with that via bunch of nested for loops
    for node in range(n_genes):
        for _ in range(number_of_functions[node]):
            line = next(env_file)
            line_no += 1
            predictors = n_predictors[fun_id]

            truth_table = np.zeros((predictors + 1, 2 ** predictors))
            truth_table[predictors] = [float(i) for i in line.split()]

            # i'm not sure if this order is the same as the one used in matlab
            for j, state in enumerate(itertools.product([0, 1], repeat=predictors)):
                for i in range(predictors):
                    truth_table[i][j] = state[i]

            truth_tables[node].append(truth_table)
            fun_id += 1


    predictor_sets = defaultdict(list)
    # get predictor sets of genes
    # some genes have more than one function - we deal with that via bunch of nested for loops
    set_id = 0
    for node in range(n_genes):
        for _ in range(number_of_functions[node]):
            line = next(env_file)
            line_no += 1
            predictor_sets[node].append([f"x{i}" for i in line.split()])

    probas = defaultdict(list)
    for node in range(n_genes):
        line = next(env_file)
        line_no += 1
        probas[node] = [float(i) for i in line.split()]

    line = next(env_file)
    line_no += 1
    perturbation_rate = float(line)

    line = next(env_file)
    line_no += 1


log_funcs = defaultdict(list)

for gen in truth_tables:
    lf = []
    for i, truth_table in enumerate(truth_tables[gen]):
        IDs = predictor_sets[gen][i]
        minterms = [list(x)[:-1] for x in truth_table.T if list(x)[-1]]

        if len(IDs) == 1:
            sym = (symbols(",".join(IDs)), )
        else:
            sym = symbols(",".join(IDs))

        fun = str(SOPform(sym, minterms, []))
        fun = translate(sym, fun)
        item = (fun, probas[gen][i])
        log_funcs[gen].append(item)

genes = [f"x{i}" for i in range(n_genes)]


# Load env
# DRUG-SYNERGY-PREDICTION
# from https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2020.00862/full
env = gym.make(f"gym-PBN/PBNEnv",
               N=args.size,
               genes=genes,
               logic_functions=log_funcs)

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
