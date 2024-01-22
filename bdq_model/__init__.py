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

from .network import DuelingNetwork, BranchingQNetwork
# from .utils import ExperienceReplayMemory

from .memory import ExperienceReplay, PrioritisedER, Transition
import utils


class BranchingDQN(nn.Module):

    def __init__(self, observation, ac, config, env):
        super().__init__()

        self.EPSILON = config.epsilon_start
        self.env = env
        self.bins = config.bins
        self.state_size, self.target_size = observation

        self.action_count = ac

        assert self.action_count == self.state_size + 1

        self.q = BranchingQNetwork(observation, ac, config.bins).to(device=config.device)
        self.target = BranchingQNetwork(observation, ac, config.bins).to(device=config.device)

        self.target.load_state_dict(self.q.state_dict())

        # model_path = 'models/laptop1_pbn28_backprop_reward/bdq_final.pt'
        # self.load_state_dict(torch.load(model_path))

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

    def dst(self, l1, l2):
        ret = 0
        for x, y in zip(l1, l2):
            if x != y:
                ret += 1
        return ret

    def predict(self, state, target):
        with torch.no_grad():
            # exploration probability
            epsilon = self.decrement_epsilon()

            # explore using edit distance
            if np.random.random() < epsilon:
                potential_actions = [np.random.randint(0, self.action_count, size=self.config.bins) for _ in range(1)]
                best_action = potential_actions[0]
                best_distance = len(best_action)

                for potential_action in potential_actions:
                    new_state = list(state)
                    for intervention in np.unique(potential_action):
                        if intervention > 0:
                            new_state[intervention-1] = 1 - new_state[intervention-1]

                    if self.dst(new_state, target) < best_distance:
                        best_action = potential_action
                        best_distance = self.dst(new_state, target)

                action = torch.tensor(best_action, device=self.config.device)
            else:
                s = np.stack((state, target))
                x = torch.tensor(s, dtype=torch.float, device=self.config.device).unsqueeze(1)

                out = self.q(x).squeeze(0)
                action = torch.argmax(out, dim=1).to(self.config.device)

            return action

    def update_policy(self, adam, memory, batch_size):
        x = memory.sample(batch_size)
        b_states, b_targets, b_actions, b_rewards, b_next_states, b_masks = zip(*x)

        states = torch.tensor(np.stack(b_states), device=self.config.device).float()
        targets = torch.tensor(np.stack(b_targets), device=self.config.device).float()
        actions = torch.stack(b_actions).long().reshape(states.shape[0], -1, 1)
        rewards = torch.tensor(np.stack(b_rewards), device=self.config.device).float().reshape(-1, 1)
        next_states = torch.tensor(np.stack(b_next_states), device=self.config.device).float()
        masks = torch.tensor(np.stack(b_masks), device=self.config.device).float().reshape(-1, 1)

        input_tuples = torch.stack((states, targets))
        qvals = self.q(input_tuples)

        current_q_values = qvals.gather(2, actions).squeeze(-1)

        with torch.no_grad():
            next_input_tuple = torch.stack((next_states, targets))
            argmax = torch.argmax(self.q(next_input_tuple), dim=2)

            max_next_q_vals = self.target(next_input_tuple).gather(2, argmax.unsqueeze(2)).squeeze(-1)

        expected_q_vals = rewards + max_next_q_vals * self.gamma * masks
        loss = F.mse_loss(expected_q_vals, current_q_values)
        self.wandb.log({"loss": loss.data})

        adam.zero_grad()
        loss.backward()

        for p in self.q.parameters():
            p.grad.data.clamp_(-1., 1.)
        adam.step()

        self.update_counter += 1
        if self.update_counter % self.target_net_update_freq == 0:
            self.update_counter = 0

            for key in self.target.state_dict():
                self.target.state_dict()[key] /= 2
                self.target.state_dict()[key] += self.q.state_dict()[key] / 2

    def decrement_epsilon(self):
        """Decrement the exploration rate."""
        self.time_steps += 1

        if self.time_steps > self.start_predicting:
            self.EPSILON = max(self.MIN_EPSILON, self.EPSILON - self.EPSILON_DECREMENT)

        return self.EPSILON

    def learn(self,
              env,
              path,
              wandb,
              ):

        config = self.config
        memory = ExperienceReplay(config.memory_size)
        adam = optim.Adam(self.q.parameters(), lr=config.learning_rate)
        self.wandb = wandb

        (state, target), _ = env.reset()
        ep_reward = 0.
        ep_len = 0
        recap = []
        rew_recap = []
        len_recap = []

        p_bar = tqdm(total=config.time_steps)
        missed = defaultdict(int)
        transitions = []

        for frame in range(config.time_steps):

            action = self.predict(state, target)

            env_action = list(action.unique())
            new_state, reward, terminated, truncated, infos = env.step(env_action)

            if truncated:
                missed[(self.env.state_attractor_id, self.env.target_attractor_id)] += 1

            if len(self.env.all_attractors) > self.attractor_count:
                self.attractor_count = len(self.env.all_attractors)
                self.EPSILON = max(self.EPSILON, 0.2)

            done = terminated | truncated
            ep_len += 1

            memory.store(Transition(
                state,
                target,
                action,
                reward,
                new_state,
                done
            ))

            if done:
                # we need to propagate reward along whole path
                ep_reward = reward

                # noinspection PyTypeChecker
                env.rework_probas(ep_len)
                (new_state, target), _ = env.reset()

                recap.append(ep_reward)
                p_bar.set_description('Rew: {:.3f}'.format(ep_reward))
                rew_recap.append(ep_reward)
                len_recap.append(ep_len)
                wandb.log({"episode_len": ep_len,
                           "episode_reward": ep_reward})
                ep_reward = 0.
                ep_len = 0

            state = new_state

            p_bar.update(1)

            if frame > max(config.batch_size, config.learning_starts):
                self.update_policy(adam, memory, config.batch_size)

            if frame % 1000 == 0:
                print(missed)
                print(f"Average episode reward: {np.average(rew_recap)}")
                print(f"Avg len: {np.average(len_recap)}")

                wandb.log({"Avg episode reward": np.average(rew_recap),
                           "Avg episode length": np.average(len_recap),
                           "Attracting state count": self.attractor_count,
                           "Exploration probability": self.EPSILON,
                           "Missed paths": len(missed)})

                # env.env.evn.env.rework_probas_epoch(len_recap)
                missed.clear()
                rew_recap = []
                len_recap = []
                self.save(f"{path}/bdq_{frame}.pt")
        self.save(f"{path}/bdq_final.pt")

    def save(self, path):
        print(path)
        parent = Path(path).parent
        parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
