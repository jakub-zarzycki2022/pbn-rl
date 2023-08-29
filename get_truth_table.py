from collections import defaultdict

import numpy as np
from itertools import product
from scipy.stats import logistic

from sympy import symbols
from sympy.logic import SOPform

import networkx as nx
import matplotlib.pyplot as plt

import gymnasium as gym
import gym_PBN

from jinja2 import Environment, FileSystemLoader, select_autoescape
jinja_env = Environment(
    loader=FileSystemLoader(searchpath="./"),
    autoescape=select_autoescape()
)

# Load env
env = gym.make("gym-PBN/Bittner-70")

g = env.env.env.env.graph

truth_tables = defaultdict(list)


def translate(logic_function):
    #print(f"got {logic_function}")
    logic_function = logic_function.replace('(', "( ")
    tokens = logic_function.split(" ")
    for i, token in enumerate(tokens):
        # ~ -> not
        if token[0] == '~':
            tokens[i] = f"~ x{token[1:]}"
        # | -> or
        elif token[0] == '|':
            tokens[i] = f"| {token[1:]}"
        # & -> and
        elif token[0] == '&':
            tokens[i] = f"& {token[1:]}"
        # skip '('
        elif token == '(':
            pass
        # {num} -> x{num}
        else:
            tokens[i] = f"x{token}"
    res = " ".join(tokens)
    #print(f"returning {res}")
    return res


for node in g.nodes:
    predictors = node.predictors

    for predictor in predictors:
        IDs, A, _ = predictor
        # matrix of # of len(state) + 1 x # of states
        truth_table = np.zeros((3 + 1 + 1, 2 ** (3 + 1)))
        for j, state in enumerate(product([0, 1], repeat=3+1)):
            x = np.ones(3 + 1)
            for i in range(len(state)):
                x[i] = state[i]
                truth_table[i][j] = state[i]
            truth_table[3+1][j] = 1 if logistic.cdf(np.dot(state, A)) >= .5 else 0
        # print(IDs)
        # print(A)
        # print(truth_table)
        truth_tables[node.ID].append((IDs, truth_table))

log_funcs = defaultdict(list)

for gen in truth_tables:
    tts = truth_tables[gen]
    lf = []
    for IDs, tt in tts:
        minterms = [list(x)[:-1] for x in tt.T if list(x)[-1]]
        pred_ids = list(IDs)
        pred_ids.append(gen)
        sym = symbols(",".join([str(x) for x in pred_ids]))
        fun = str(SOPform(sym, minterms, []))
        if fun == 'True':
            fun = f'{gen} | ~{gen}'
        lf.append((translate(fun)))
    log_funcs[gen] = lf


# print("\n\n\n")
#
# for key in log_funcs:
#     print(f"x{key}: boolean;")
#
# print("")
# for key in log_funcs:
#     for fun in log_funcs[key]:
#         print(f"x{key}=true if ({fun})=true;")
#         print(f"x{key}=false if ({fun})=false;")

template = jinja_env.get_template("model_template.jj2")

out = template.render(log_funcs=log_funcs)

with open("model_from_jinja.ispl", "w+") as f:
    f.write(out)
