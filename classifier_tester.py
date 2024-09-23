import argparse
import random
import itertools

from collections import defaultdict
import gymnasium as gym
import gym_PBN
import torch
from gym_PBN.envs.bittner.base import findAttractors

from graph_classifier.classifier_agent import ClassifierAgent

import math

from gbdq_model.utils import ExperienceReplayMemory, AgentConfig


import seaborn as sns
from matplotlib import pyplot as plt

from gym_PBN.utils.get_attractors_from_cabean import get_attractors

# parser = argparse.ArgumentParser()
# parser.add_argument('-n', type=int, required=True)
# parser.add_argument('--model-path', required=True)
# parser.add_argument('--attractors', type=int, default=3)
# args = parser.parse_args()

model_cls = ClassifierAgent
model_name = "ClassifierAgent"

N = 28
model_path = "models/laptop_tmp_pbn28_classifier/bdq_final.pt"
min_attractors = 16
# model_path = args.model_path
# min_attractors = args.attractors

env = gym.make("gym-PBN/BittnerMultiGeneral", N=N, min_attractors=min_attractors)
env.reset()

DEVICE = 'cpu'

config = AgentConfig()
model = model_cls(N, min_attractors, config, env)
model.load_state_dict(model_path)

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

for i in range(1):
    print("testing round ", i)
    for attractor_id, target_id in itertools.product(range(len(env.all_attractors)), repeat=2):
        print(f"processing initial_state, target_state = {attractor_id}, {target_id}")
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

            if count > 10:
                print(f"failed to converge for {attractor_id}, {target_id}")
                #print(f"final state was {tuple(state)}")
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


    print(f"{failed} failed states out of {total}")

print(f"the avg is {sum(lens) / len(lens)} with len: {len(lens)}")

data = defaultdict(int)
for i in lens:
    data[i] += 1

total = sum(lens)
last = max(data.keys())

x = list(range(1, last+1))

y = [math.ceil(data[i]) for i in x]

labels = [i if i % 5 == 0 and i < 20 else '' for i in x]
labels[0] = 1

for i in range(30, len(x)):
    if y[i] > 0:
        labels[i] = x[i]
        for j in range(1, 5):
            labels[i-j] = ''

d2 = {'x':x, 'y':y}
plt.figure(figsize=(20, 8))
ax = sns.barplot(data=d2, x='x', y='y', color='blue', label='big')
ax.set_xticklabels(labels)
ax.tick_params(labelsize=40)
plt.savefig(f'bn{N}.pdf', bbox_inches='tight', pad_inches=0.)

# for manual fixes
print(lens)

# sns.distplot(lens, bins="doane", kde=False, hist_kws={"align": "left"})
# plt.show()
