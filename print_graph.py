import networkx as nx
import matplotlib.pyplot as plt

import networkx as nx
from networkx.utils.decorators import not_implemented_for


import gymnasium as gym
import gym_PBN

# Load env
env = gym.make("gym-PBN/Bittner-7")


s = env.env.env.env.graph.genSTG()

adj = []
for key in s:
  _, al = s[key]
  for trg in al:
    adj.append((key, trg, al[trg]))

g = nx.DiGraph()
g.add_nodes_from(s.keys())
g.add_weighted_edges_from(adj)

scc = list(nx.strongly_connected_components(g))
cG = nx.condensation(g, scc)

attr = []

for n in cG:
  if cG.out_degree(n) == 0:
    attr.append(scc[n])

nx.draw(g)
plt.show()


import networkx as nx

predictor_sets_path = "pbn_inference/data"
genedata = f"{predictor_sets_path}/genedata.xls"
TOTAL_GENES = 7

includeIDs = [234237, 324901, 759948, 25485, 266361, 108208, 130057]

graph = spawn(
    file=genedata,
    total_genes=TOTAL_GENES,
    include_ids=includeIDs,
    bin_method="kmeans",
    n_predictors=5,
    predictor_sets_path=predictor_sets_path,
)

graph.genRandState()

s = graph.genSTG('stg.png')

adj = []
for key in s:
  _, al = s[key]
  for trg in al:
    adj.append((key, trg, al[trg]))

g = nx.DiGraph()
g.add_nodes_from(s.keys())
g.add_weighted_edges_from(adj)

scc = list(nx.strongly_connected_components(g))
cG = nx.condensation(g, scc)

attr = []

for n in cG:
  if cG.out_degree(n) == 0:
    attr.append(scc[n])

print(attr)
