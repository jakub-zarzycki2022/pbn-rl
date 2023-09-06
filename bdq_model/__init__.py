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

from .network import DuelingNetwork, BranchingQNetwork
# from .utils import ExperienceReplayMemory

from ddqn_per.memory import ExperienceReplay, PrioritisedER, Transition
import bdq_model.utils


class BranchingDQN(nn.Module):

    def __init__(self, observation, ac, config):
        super().__init__()

        self.EPSILON = 0.9
        self.bins = config.bins
        self.state_size, self.target_size = observation

        self.action_count = ac

        assert self.action_count == self.state_size + 1

        self.q = BranchingQNetwork(observation, ac, config.bins)
        self.target = BranchingQNetwork(observation, ac, config.bins)

        self.target.load_state_dict(self.q.state_dict())

        self.target_net_update_freq = config.target_net_update_freq
        self.config = config
        self.update_counter = 0

        self.MIN_EPSILON = 0.05

    def predict(self, state, target):
        with torch.no_grad():
            # a = self.q(x).max(1)[1]
            x = torch.tensor([[state], [target]], dtype=torch.float)
            out = self.q(x).squeeze(0)
            action = torch.argmax(out, dim=1)
            return list(set(action))

    def update_policy(self, adam, memory, batch_size):
        x = memory.sample(batch_size)
        b_states, b_targets, b_actions, b_rewards, b_next_states, b_masks = zip(*x)

        states = torch.tensor(b_states).float()
        targets = torch.tensor(b_targets).float()
        actions = torch.tensor(b_actions).long().reshape(states.shape[0], -1, 1)
        rewards = torch.tensor(b_rewards).float().reshape(-1, 1)
        next_states = torch.tensor(b_next_states).float()
        masks = torch.tensor(b_masks).float().reshape(-1, 1)

        input_tuple = torch.stack((states, targets))
        qvals = self.q(input_tuple)

        current_q_values = qvals.gather(2, actions).squeeze(-1)

        with torch.no_grad():
            next_input_tuple = torch.stack((states, targets))
            argmax = torch.argmax(self.q(next_input_tuple), dim=2)

            max_next_q_vals = self.target(next_input_tuple).gather(2, argmax.unsqueeze(2)).squeeze(-1)
            max_next_q_vals = max_next_q_vals.mean(1, keepdim=True)

        expected_q_vals = rewards + max_next_q_vals * 0.99 * masks
        loss = F.mse_loss(expected_q_vals, current_q_values)

        adam.zero_grad()
        loss.backward()

        for p in self.q.parameters():
            p.grad.data.clamp_(-1., 1.)
        adam.step()

        self.update_counter += 1
        if self.update_counter % self.target_net_update_freq == 0:
            self.update_counter = 0
            self.target.load_state_dict(self.q.state_dict())

    def decrement_epsilon(self):
        """Decrement the exploration rate."""
        self.EPSILON = max(self.MIN_EPSILON, self.EPSILON * 0.9)
        return self.EPSILON

    def learn(self,
              env,
              path,
              ):

        config = self.config
        memory = ExperienceReplay(config.memory_size)
        adam = optim.Adam(self.q.parameters(), lr=config.learning_rate)

        (state, target), _ = env.reset()
        ep_reward = 0.
        ep_len = 0
        recap = []
        rew_recap = []
        len_recap = []

        p_bar = tqdm(total=config.time_steps)
        for frame in range(config.time_steps):

            epsilon = self.decrement_epsilon()

            if np.random.random() > epsilon:
                action = self.predict(state, target)
            else:
                action = np.random.randint(0, self.action_count, size=config.bins)

            new_state, reward, terminated, truncated, infos = env.step(action)
            done = terminated | truncated
            ep_reward += reward
            ep_len += 1

            if done:
                (new_state, target), _ = env.reset()
                recap.append(ep_reward)
                p_bar.set_description('Rew: {:.3f}'.format(ep_reward))
                rew_recap.append(ep_reward)
                len_recap.append(ep_len)
                ep_reward = 0.
                ep_len = 0

            memory.store(Transition(
                state,
                target,
                action,
                reward,
                new_state,
                done
            ))

            # memory.push((s.reshape(-1).numpy().tolist(), action, reward, new_state.reshape(-1).numpy().tolist(), 0. if done else 1.))
            state = new_state

            p_bar.update(1)

            if frame > config.batch_size:
                self.update_policy(adam, memory, config.batch_size)

            if frame % 1000 == 0:
                print(f"Average episode reward: {np.average(rew_recap)}")
                print(f"Avg len: {np.average(len_recap)}")
                rew_recap = []
                len_recap = []
                self.save(f"{path}/bdq_{frame}.pt")
        self.save(f"{path}/bdq_final.pt")

    def save(self, path):
        print(path)
        parent = Path(path).parent
        parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
