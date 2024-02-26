from collections import defaultdict

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from pathlib import Path

import numpy as np
import gym
import random

from gym_PBN.envs import PBNTargetMultiEnv
from .network import GraphClassifierNetwork
# from .utils import ExperienceReplayMemory

from .memory import ExperienceReplay, PrioritisedER, Transition
import utils


class GraphClassifier(nn.Module):

    def __init__(self, state_size, attractors, config, env):
        super().__init__()

        self.env = env
        self.bins = config.bins
        self.attractors = attractors
        self.state_size = state_size

        self.classifier_net = GraphClassifierNetwork(state_size, attractors).to(device=config.device)

        # model_path = 'models/laptop1_pbn28_backprop_reward/bdq_final.pt'
        # self.load_state_dict(torch.load(model_path))

        self.config = config
        self.update_counter = 0

        self.time_steps = 0
        self.start_predicting = config.learning_starts

        self.edge_index = self.get_adj_list()

        self.wandb = None
        self.attractor_count = len(env.attracting_states)

    def predict(self, state):
        with torch.no_grad():
            x = torch.tensor(state, dtype=torch.float, device=self.config.device)
            x = x.t()
            # x = x.unsqueeze(dim=0)

            probas = self.q(x, self.edge_index).squeeze(0)

            return probas

    def update_policy(self, adam, memory, batch_size):

        if len(memory) == 0:
            return -1

        sample = random.sample(tuple(memory.keys()), min(batch_size, len(memory)))
        # print("memory: ", memory.keys())
        # print("sample: ", [a for a in memory])

        memory_policy = []

        for item in sample:
            policy = [memory[item][i] / sum(memory[item].values()) for i in range(self.attractors)]
            memory_policy.append(policy)

        # print(sample)
        # print(memory_policy)
        net_policy = self.classifier_net(torch.tensor(sample,
                                                      dtype=torch.float,
                                                      device=self.config.device).unsqueeze(dim=2),
                                         edge_index=self.edge_index)

        print("net: ", torch.exp(net_policy[0]))
        print("real: ", memory_policy[0])
        print("dif: ", F.kl_div(net_policy[0], torch.tensor(memory_policy[0],
                                             dtype=torch.float,
                                             device=self.config.device)))

        loss = F.kl_div(net_policy, torch.tensor(memory_policy,
                                                  dtype=torch.float,
                                                  device=self.config.device))
        self.wandb.log({"loss": loss.data})

        adam.zero_grad()
        loss.backward()
        adam.step()

        return loss

    def learn(self,
              env,
              path,
              wandb,
              ):

        config = self.config
        self.bs = config.batch_size

        # maps state -> (attractor, number of hits)
        memory = defaultdict(lambda: defaultdict(int))
        adam = optim.Adam(self.classifier_net.parameters(), lr=config.learning_rate)
        self.wandb = wandb

        (state, target), _ = env.reset()
        recap = []

        p_bar = tqdm(total=config.time_steps)

        state_size = len(state)
        history = []

        state = tuple([random.randint(0, 1) for _ in range(state_size)])
        env.graph.setState(state)
        loss = -1

        for frame in range(config.time_steps):

            history.append(state)

            if env.is_attracting_state(state):
                attractor_id = -1

                for id, a in enumerate(env.all_attractors):
                    if a[0] == state:
                        attractor_id = id

                for historical_state in history:
                    memory[historical_state][attractor_id] += 1

                history = []
                state = tuple([random.randint(0, 1) for _ in range(state_size)])
                env.graph.setState(state)
            else:
                state, _, _, _, _ = env.step([])

            if len(self.env.all_attractors) > self.attractor_count:
                self.attractor_count = len(self.env.all_attractors)
                # self.EPSILON = max(self.EPSILON, 0.2)

            # noinspection PyTypeChecker
            env.rework_probas()

            p_bar.update(1)

            if frame % config.batch_size == 0:
                loss = self.update_policy(adam, memory, config.batch_size)

            p_bar.set_description(f'Loss: {loss}')

            if frame % 1000 == 0:

                wandb.log({
                           "Attracting state count": self.attractor_count,
                })

                self.save(f"{path}/bdq_{frame}.pt")
        self.save(f"{path}/bdq_final.pt")

    def save(self, path):
        print(path)
        parent = Path(path).parent
        parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def get_adj_list(self):
        env = self.env
        top_nodes = []
        bot_nodes = []

        for top_node in env.graph.nodes:
            done = set()
            top_nodes.append(top_node.index)
            bot_nodes.append(top_node.index)

            for predictor, _, _ in top_node.predictors:
                for bot_node_id in predictor:
                    if bot_node_id not in done:
                        done.add(bot_node_id)
                        top_nodes.append(top_node.index)
                        bot_nodes.append(env.graph.getNodeByID(bot_node_id).index)

        return torch.tensor([top_nodes, bot_nodes], dtype=torch.long, device=self.config.device)

