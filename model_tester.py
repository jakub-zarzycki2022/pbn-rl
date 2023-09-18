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

import wandb
from ddqn_per import DDQNPER, DDQN
from bdq_model import BranchingDQN

from bdq_model.utils import ExperienceReplayMemory, AgentConfig

model_cls = BranchingDQN
model_name = "BranchingDQN"

env = gym.make("gym-PBN/BittnerMulti-28")
env.reset()

DEVICE = 'cpu'
model_path = 'models/jz_v13_pbn28_multi/bdq_final.pt'; 

config = AgentConfig()
model = BranchingDQN((28, 28), 29, config)
model.load_state_dict(torch.load(model_path))


action = 0
(state, target), _ = env.reset()
action = model.predict(state, target); print(action); state, *_ = env.step(action); print(state)

lens = []
all_attractors = env.env.env.env.all_attractors
gen_ids = env.env.env.env.includeIDs

failed = 0
total = 0

failed_pairs = []

for attractor_id, target_id in itertools.product(range(len(all_attractors)), repeat=2):
    #print(f"processing initial_state, target_state = {attractor_id}, {target_id}")
    attractor = all_attractors[attractor_id]
    target = all_attractors[target_id]
    
    _ = env.env.env.env.setTarget(target)
    target_state = [0 if i == '*' else i for i in random.choice(target)]
    for initial_state in attractor:
        total += 1
        actions = []
        state = initial_state
        state = [0 if i == '*' else i for i in list(state)]
        _ = env.reset()
        _ = env.env.env.env.graph.setState(state)
        count = 0
        while not env.in_target(state):
            count += 1
            action = model.predict(state, target_state)
            action = action.unique().tolist()
            _ = env.step(action)
            state = env.render()
            action_named = [gen_ids[a-1] for a in action]
            actions.append(action_named)
            if count > 100:
                print(f"failed to converge for initial state {initial_state}")
                #print(f"final state was 		     {tuple(state)}")
                failed += 1
                failed_pairs.append((initial_state, target))
                break
        else:
            print(f"for initial state {initial_state} and target {target} got (total of {count} steps)")
            for a in actions:
               #print(a)
               pass
            lens.append(count)
            
        print(f"{failed} failed states out of {total}")
        
        
for x, y in failed_pairs:
  print(x, y)
  

