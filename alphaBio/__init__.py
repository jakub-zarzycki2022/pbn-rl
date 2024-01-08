from collections import defaultdict, deque
from itertools import product

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path

import numpy as np
import gym
import random

from .network import NNet
from .MCTS import MCTS


class AlphaBio(nn.Module):

    def __init__(self, observation, ac, config, env):
        super().__init__()

        self.EPSILON = config.epsilon_start
        self.env = env
        self.bins = config.bins
        self.state_size, self.target_size = observation

        self.action_count = ac

        assert self.action_count == self.state_size + 1

        self.nnet = NNet(observation, ac, config.bins).to(device=config.device)

        self.target_net_update_freq = config.target_net_update_freq
        self.config = config
        self.gamma = config.gamma
        self.update_counter = 0

        self.time_steps = 0
        self.start_predicting = config.learning_starts
        self.reward_discount_rate = config.reward_discount_rate

        self.MIN_EPSILON = config.epsilon_final
        self.MAX_EPSILON = config.epsilon_start
        self.EPSILON_DECREMENT = (self.MAX_EPSILON - self.MIN_EPSILON) / config.epsilon_decay

        self.wandb = None

        self.attractor_count = len(env.attracting_states)
        self.edge_index = self.get_adj_list()
        print(self.edge_index)

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

    def dst(self, l1, l2):
        ret = 0
        for x, y in zip(l1, l2):
            if x != y:
                ret += 1
        return ret

    def predict(self, state, target):
        self.nnet.eval()
        with torch.no_grad():
            x = torch.tensor((state, target), dtype=torch.float, device=self.config.device)
            x = x.t()
            x = x.unsqueeze(dim=0)

            policy, value = self.nnet(x, self.edge_index)

            return torch.exp(policy).data[0], value.data[0]

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def update_policy(self, adam, memory):

        b_states, b_targets, b_policies, b_values = list(zip(*memory))

        states = torch.tensor(b_states, device=self.config.device, dtype=torch.float)
        targets = torch.tensor(b_targets, device=self.config.device, dtype=torch.float)
        target_policies = torch.tensor(b_policies, device=self.config.device, dtype=torch.float)
        target_values = torch.tensor(b_values, device=self.config.device, dtype=torch.float)

        input_tuples = torch.stack((states, targets), dim=2)

        out_pi, out_v = self.nnet(input_tuples, self.edge_index)

        l_pi = self.loss_pi(target_policies, out_pi)
        l_v = self.loss_v(target_values, out_v)
        loss = l_pi + l_v

        self.wandb.log({"loss": loss.data})

        adam.zero_grad()
        loss.backward()

        # for p in self.q.parameters():
        #     p.grad.data.clamp_(-1., 1.)

        adam.step()

    def decrement_epsilon(self):
        """Decrement the exploration rate."""
        self.time_steps += 1

        if self.time_steps > self.start_predicting:
            self.EPSILON = max(self.MIN_EPSILON, self.EPSILON - self.EPSILON_DECREMENT)

        return self.EPSILON

    def execute_episode(self, mcts: MCTS):
        (state, target), _ = self.env.reset()
        print(state)
        print(target)
        self.config.tempThreshold = 1
        trainExamples = []
        episodeStep = 0
        # state = self.env.render()
        # target = self.env.target[0]

        while True:
            episodeStep += 1
            temp = int(episodeStep < self.config.tempThreshold)

            pi = mcts.get_action_prob(state, target, temp=1)
            trainExamples.append([state, target, pi, None])

            action = np.random.choice(len(pi), p=pi)
            state, reward, terminated, truncated, info = self.env.step([action])

            done = terminated | truncated

            if done:
                gamma = self.config.reward_discount_rate
                for i in range(len(trainExamples)):
                    trainExamples[i][3] = 0 if truncated else 1  # else reward ** (episodeStep - i)
                return trainExamples

    def learn(self,
              env,
              path,
              wandb,
              ):

        config = self.config

        adam = optim.Adam(self.nnet.parameters(), lr=config.learning_rate)
        self.wandb = wandb

        rew_recap = []
        len_recap = []

        p_bar = tqdm(total=config.num_iteration)
        missed = defaultdict(int)

        for i in range(config.num_iteration):
            iterationTrainExamples = []

            while len(iterationTrainExamples) < config.batch_size:
                mcts = MCTS(env, self)  # reset search tree
                episode = self.execute_episode(mcts)
                iterationTrainExamples.extend(episode)

                ep_reward = episode[-1][3]
                ep_len = len(episode)
                p_bar.set_description('Rew: {:.3f}'.format(ep_reward))
                rew_recap.append(ep_reward)
                len_recap.append(ep_len)
                wandb.log({"episode_len": ep_len,
                           "episode_reward": ep_reward})

                p_bar.update(1)

            # training new network, keeping a copy of the old one
            self.save(f"{path}/bdq_{i}.pt")

            random.shuffle(iterationTrainExamples)

            self.update_policy(adam, iterationTrainExamples)

            if i % 100 == 0:
                print(missed)
                print(f"Average episode reward: {np.average(rew_recap)}")
                print(f"Avg len: {np.average(len_recap)}")

                wandb.log({"Avg episode reward": np.average(rew_recap),
                           "Avg episode length": np.average(len_recap),
                           "Attracting state count": self.attractor_count,
                           "Exploration probability": self.EPSILON,
                           "Missed paths": sum(missed.values())})

                # env.rework_probas_epoch(len_recap)
                missed.clear()
                rew_recap = []
                len_recap = []
                self.save(f"{path}/alphabio_{i}.pt")
            self.save(f"{path}/alphabio_final.pt")

    def save(self, path):
        parent = Path(path).parent
        parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
