import argparse
import random
from pathlib import Path
import itertools

from collections import defaultdict
import gymnasium as gym
import gym_PBN
import numpy as np
import torch
from gym_PBN.utils.eval import compute_ssd_hist
from gym_PBN.envs.bittner.base import findAttractors

from alphaBio import AlphaBio
from bdq_model import BranchingDQN


from alphaBio.utils import ExperienceReplayMemory, AgentConfig
from alphaBio.MCTS import MCTS

import seaborn as sns
from matplotlib import pyplot as plt

from gym_PBN.utils.get_attractors_from_cabean import get_attractors

parser = argparse.ArgumentParser()
parser.add_argument('-n', type=int, required=True)
parser.add_argument('--model-path', required=True)
parser.add_argument('--attractors', type=int, default=3)
args = parser.parse_args()

model_cls = BranchingDQN
model_name = "BranchingDQN"

N = args.n
model_path = args.model_path
min_attractors = args.attractors

env = gym.make("gym-PBN/BittnerMultiGeneral", N=N, min_attractors=min_attractors)
env.reset()


DEVICE = 'cpu'


config = AgentConfig()
model = model_cls((N, N), N+1, config, env)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))


action = 0
(state, target), _ = env.reset()

# policy, value = model.predict(state, target);
# policy = policy.numpy()
# action = [np.random.choice(range(N+1), p=policy)]

action = model.predict(state, target)
print(action)
state, *_ = env.step(action)
print(state)


all_attractors = env.all_attractors

lens = []
failed = 0
total = 0

failed_pairs = []

all_attractors = env.all_attractors
print("genereted attractors:")
for a in all_attractors:
    print(a)

lens = []
failed = 0
total = 0

failed_pairs = []

for i in range(10):
    print("testing round ", i)
    for attractor_id, target_id in itertools.product(range(len(all_attractors)), repeat=2):
        #print(f"processing initial_state, target_state = {attractor_id}, {target_id}")
        attractor = all_attractors[attractor_id]
        target = all_attractors[target_id]
        target_state = target[0]
        initial_state = attractor[0]
        total += 1
        actions = []
        state = initial_state
        state = [0 if i == '*' else i for i in list(state)]
        _ = env.reset()
        env.graph.setState(state)
        count = 0
        
        env.setTarget(target)

        while not env.in_target(state):
            count += 1
            
            # policy, value = model.predict(state, target_state)
            # policy = policy.numpy()
            # action = [np.random.choice(range(N+1), p=policy)]
            action = model.predict(state, target_state)
            
            _ = env.step(action)
            state = env.render()
            #action_named = [gen_ids[a-1] for a in action]

            if count > 1000:
                print(f"failed to converge for {attractor_id}, {target_id}")
                #print(f"final state was 		     {tuple(state)}")
                failed += 1
                failed_pairs.append((initial_state, target))
                #raise ValueError
                break
        else:
            print(f"for initial state {attractor_id} and target {target_id} got (total of {count} steps)")
            #raise ValueError()
            for a in actions:
               #print(a)
               pass
            if count > 0:
                lens.append(count)


    print(np.average(lens))        
    print(f"{failed} failed states out of {total}")


for x, y in failed_pairs:
    print(x, y)

sns.distplot(lens, bins="doane", kde=False, hist_kws={"align": "left"})
plt.show()
