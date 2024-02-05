import logging
import math
from collections import defaultdict
import torch

import numpy as np

EPS = 1e-8


# TODO: Make the search parallel
# TODO: Add more variation to the reward / value function
class MCTS:
    """
    This class handles the MCTS tree.
    """

    def __init__(self, env, model):
        self.env = env
        self.model = model
        self.cpuct = 1
        self.args = None
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = defaultdict(int)  # stores #times edge s,t,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s

    def get_action_prob(self, state, target, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,t,a)]**(1./temp)
        """
        for i in range(5):
            self.search(state, target, 5)

        s = tuple(state)
        t = tuple(target)
        counts = [self.Nsa[(s, t, a)] for a in range(len(s) + 1)]

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, state, target, max_depth=20):
        if max_depth <= 0:
            return -1

        max_depth -= 1

        state = tuple(state)
        target = tuple(target)

        if state == target:
            self.Es[(state, target)] = 1
            return 1

        if (state, target) not in self.Ps:
            # leaf node
            self.Ps[(state, target)], v = self.model.predict(state, target)
            sum_Ps_s = torch.sum(self.Ps[(state, target)])
            self.Ps[(state, target)] /= sum_Ps_s  # renormalize

            self.Ns[(state, target)] = 0
            # print("s not in Ps")
            return v

        cur_best = float('-inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(len(state) + 1):
            if (state, target, a) in self.Qsa:
                u = self.Qsa[(state, target, a)] + \
                    self.cpuct * \
                    self.Ps[(state, target)][a] * \
                    math.sqrt(self.Ns[(state, target)]) / \
                    (1 + self.Nsa[(state, target, a)])
            else:
                u = self.cpuct * self.Ps[(state, target)][a] * math.sqrt(self.Ns[(state, target)] + EPS)

            if u > cur_best:
                cur_best = u
                best_act = a

        a = best_act
        next_state = self.env.get_next_state(state, [a])

        # if next_state == state:
        #     max_depth /= 2

        v = self.search(next_state, target, max_depth)

        if (state, target, a) in self.Qsa:
            self.Qsa[(state, target, a)] = (self.Nsa[(state, target, a)] * self.Qsa[(state, target, a)] + v) / \
                                           (self.Nsa[(state, target, a)] + 1)
            self.Nsa[(state, target, a)] += 1
        else:
            self.Qsa[(state, target, a)] = v
            self.Nsa[(state, target, a)] = 1

        self.Ns[(state, target)] += 1
        return v
