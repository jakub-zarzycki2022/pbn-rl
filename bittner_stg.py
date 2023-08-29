import argparse
import random
from pathlib import Path

import gymnasium as gym
import gym_PBN
import numpy as np
import torch
from gym_PBN.utils.eval import compute_ssd_hist

import networkx as nx
import matplotlib.pyplot as plt

import wandb
from ddqn_per import DDQNPER
model_cls = DDQNPER
model_name = "DDQNPER"


# Load env
env = gym.make("gym-PBN/Bittner-7")

graph = env.env.env.env.unwrapped.graph

s = nx.DiGraph()
stg = graph.genSTG()

s.add_nodes_from(stg.keys())

for node in stg:
  edges = stg[node][1]
  for edge in edges:
    s.add_edge(node, edge, weight=edges[edge])

for scc in nx.algorithms.components.attracting_components(s):
   print(scc)

pos = nx.spring_layout(s) # pos = nx.nx_agraph.graphviz_layout(G)
nx.draw(s, pos)
labels = nx.get_edge_attributes(s, 'weight')
nx.draw(s, with_labels=True)
plt.show()
