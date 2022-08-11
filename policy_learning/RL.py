#!/usr/bin/python
import numpy as np
import random
from datetime import datetime


## TODO
# tabular, Qlearning, Sarsa, eligibility traces, actor critic, policy gradient, Q or V, QNN

# Return list of position of largest element  -- RANDOM between equals
def maxs(seq):
    max_indices = []
    # if seq:
    max_val = seq[0]
    for i, val in ((i, val) for i, val in enumerate(seq) if val >= max_val):
        if val == max_val:
            max_indices.append(i)
        else:
            max_val = val
            max_indices = [i]
    return random.choice(max_indices)


def cdf(seq):
    r = random.random()
    print(seq, r)
    for i, s in enumerate(seq):
        if r <= s:
            return i


class MDP:
    def __init__(self, init, actlist, terminals=[], gamma=.9):
        self.init = init
        self.actlist = actlist
        self.terminals = terminals
        if not (0 <= gamma < 1):
            raise ValueError("An MDP must have 0 <= gamma < 1")
        self.gamma = gamma
        self.states = set()
        self.reward = 0

    def actions(self, state):
        """Set of actions that can be performed in this state.  By default, a
        fixed list of actions, except for terminal states. Override this
        method if you need to specialize by state."""
        if state in self.terminals:
            return [None]
        else:
            return self.actlist


class Policy:
    def __init__(self, name, param, p_guidance_mistakes=None, Q_state=[]):
        self.name = name
        self.param = param
        self.Q_state = Q_state
        if name == 'guidance':
            if p_guidance_mistakes > 1 or p_guidance_mistakes < 0:
                raise ValueError("Probability of guidance error has to be value between 0 and 1")
            self.p_guidance_mistakes = p_guidance_mistakes

    def return_action(self):
        if self.name == 'softmax':
            values = self.Q_state
            tau = self.param
            maxQ = 0
            av = np.asarray(values)
            n = len(av)
            probs = np.zeros(n)

            for i in range(n):
                softm = (np.exp(av[i] / tau) / np.sum(np.exp(av[:] / tau)))
                probs[i] = softm
            prob = np.random.multinomial(1, probs)
            arg = np.where(prob)[0][0]
            return arg

        elif self.name == 'guidance':
            self.mistake = False
            values = self.Q_state
            if_correct = random.random()
            if  if_correct <= 1-self.p_guidance_mistakes:
                self.mistake = False
                arg = maxs(values)
            else:
                self.mistake = True
                actions_to_sample = np.arange(stop=len(values))
                action_to_remove = maxs(values)
                actions_to_sample = np.delete(actions_to_sample, maxs(values))
                arg = actions_to_sample[np.random.randint(len(actions_to_sample), size=1)[0]]
            return arg

        elif self.name == 'egreedy':
            raise NotImplementedError("Implementation of egreedy policy has to be corrected")
            values = self.Q_state
            maxQ = max(values)
            e = self.param
            if random.random() < e:  # exploration
                return random.randint(0, len(values) - 1)
            else:  # exploitation
                return maxs(values)

        elif self.name == 'exploitation':
            values = self.Q_state
            return maxs(values)

        else:
            raise NotImplementedError


class Representation:
    # qtable, neural network, policy function, function approximation
    def __init__(self, name, params):
        self.name = name
        self.params = params
        if self.name == 'qtable':
            [self.actlist, self.states] = self.params
            self.Q = [[0.0] * len(self.actlist) for x in range(len(self.states))]


class Learning:
    # qlearning, sarsa, traces, actor critic, policy gradient
    def __init__(self, name, params):
        self.name = name
        self.params = params
        if self.name == 'qlearn' or self.name == 'sarsa':
            self.alpha = self.params[0]
            self.gamma = self.params[1]

    def update(self, state, action, next_state, next_action, reward, Q_state, Q_next_state, done):
        if done:
            Q_state[action] = Q_state[action] + self.alpha * (reward - Q_state[action])
            error = reward - Q_state[action]
        else:
            if self.name == 'qlearn':
                add = reward + self.gamma * max(Q_next_state) - Q_state[action]
                Q_state[action] += self.alpha * (reward + self.gamma * max(Q_next_state) - Q_state[action])
                error = reward + self.gamma * max(Q_next_state) - Q_state[action]

            if self.name == 'sarsa':
                learning = self.alpha * (reward + self.gamma * Q_next_state[next_action] - Q_state[action])
                Q_state[action] = Q_state[action] + learning
                error = reward + self.gamma * Q_next_state[next_action] - Q_state[action]
        return Q_state, error


"""
m = MDP([0,0], [0,1,2,3,4], 1)
m.states = [[0,0], [0,1], [1,1]]
print m.actlist, m.states
q = Representation('qtable', [m.actlist, m.states])
print q.Q
softmax = Policy('softmax', 1, [11.2, 2.2, 13.4, 12.3])
a = softmax.return_action()
print a
egreedy = Policy('egreedy', 0, [1.2, 2.2, 13.4, 12.3])
a = egreedy.return_action()
print a
"""
