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

from alphaBio.utils import ExperienceReplayMemory, AgentConfig
from alphaBio.MCTS import MCTS

import seaborn as sns
from matplotlib import pyplot as plt

model_cls = AlphaBio
model_name = "AlphaBio"

N = 7
env = gym.make("gym-PBN/BittnerMultiGeneral", N=N)
#env = gym.make("gym-PBN/BittnerMulti-28")
env.reset()


DEVICE = 'cpu'

#model_path = 'models/laptop1_pbn28_backprop_reward/bdq_final.pt'; 
#model_path = 'models/pbn10_pbn10_bdq/bdq_final.pt'; 
#path_model = 'models/for_paper_new_arch_pbn28_cluster/bdq_100000.pt';
#model_path = 'models/jz_v3_pbn7_for_paper//bdq_final.pt';
#model_path = 'models/cluster9_pbn28_256input_complicated/bdq_247000.pt'

# works well
# model_path = 'models/laptop2_pbn7_256input_neg_reward/alphaBio_36000.pt'

model_path = "models/laptop3_pbn7_256input_neg_reward/alphaBio_15000.pt"

#model_path = args.model_path

config = AgentConfig()
model = AlphaBio((N, N), N+1, config, env)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))


action = 0
(state, target), _ = env.reset()
policy, value = model.predict(state, target); 
policy = policy.numpy()
action = [np.random.choice(range(N+1), p=policy)]

	
print(action); 
state, *_ = env.step(action); 
print(state)

lens = []
all_attractors = env.all_attractors

failed = 0
total = 0

failed_pairs = []

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
        policy, value = model.predict(state, target_state)
        policy = policy.numpy()
        action = [np.random.choice(range(N+1), p=policy)]

        _ = env.step(action)
        state = env.render()
        #action_named = [gen_ids[a-1] for a in action]

        if count > 100:
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
        lens.append(count)

all_attractors = env.all_attractors
print("genereted attractors:")
for a in all_attractors:
  print(a)
  
print("and")
for a in env.attracting_states:
  print(a)
  
print("\n\nround 2")

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
        
        policy, value = model.predict(state, target_state)
        policy = policy.numpy()
        action = [np.random.choice(range(N+1), p=policy)]
        
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
        lens.append(count)


print(np.average(lens))        
print(f"{failed} failed states out of {total}")


for x, y in failed_pairs:
    print(x, y)

sns.distplot(lens, bins="doane", kde=False, hist_kws={"align": "left"})
plt.show()
