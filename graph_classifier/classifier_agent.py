from itertools import combinations

import gym as gym
import gym_PBN
import torch

from . import GraphClassifier
from .utils import AgentConfig


class ClassifierAgent:
    def __init__(self, N=36, attractors=40, config=None, env=None):
        self.N = N
        self.attractor_count = attractors
        self.config = config
        self.env = env
        self.model = GraphClassifier(N, attractors, self.config, self.env)

    def state_to_tensor(self, state):
        x = torch.tensor(state, dtype=torch.float)
        return x.reshape(1, self.N, 1)

    def get_best_id(self, state, target_id):
        max_p = 0
        max_id = -1
        for comb in combinations(range(self.N), 3):
            ns = list(state)
            for i in comb:
                ns[i] = 1 - ns[i]
            x = self.state_to_tensor(ns)
            y = torch.exp(self.model.classifier_net(x, self.model.edge_index)[0, target_id])
            if y > max_p:
                max_p = y
                max_id = comb
        return max_p, max_id

    def predict(self, state, target):
        target_id = self.attractor_id(target)
        p, actions = self.get_best_id(state, target_id)

        return list(actions)

    def attractor_id(self, state):
        for i in range(len(self.env.all_attractors)):
            if self.env.all_attractors[i][0] == tuple(state):
                return i
        return -1

    def load_state_dict(self, path):
        self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
