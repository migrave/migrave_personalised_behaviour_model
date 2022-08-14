#!/usr/bin/python

'''
    Copyright 2022 by Michał Stolarz <michal.stolarz@h-brs.de>

    This file is part of migrave_personalised_behaviour_model,
    and is based on: https://github.com/TsiakasK/sequence-learning-dataset.
    It contans all the classes needed for representing a Reinforcement Learning problem.

    migrave_personalised_behaviour_model is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    migrave_personalised_behaviour_model is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.
    You should have received a copy of the GNU Affero General Public License
    along with migrave_personalised_behaviour_model. If not, see <http://www.gnu.org/licenses/>.
'''

import numpy as np
import random
from datetime import datetime

from behaviour_model.utils import maxs


class Policy:
    def __init__(self, name, param, p_guidance_mistakes=None, Q_state=[]):
        """
        Class representing an exploration policy.
        :param name: type of the exploration strategy
        (guidance, softmax, egreedy or pure exploitation (no exploration))
        :param param: parameter defining the level of randomness for an exploration
        :param p_guidance_mistakes: probability of supervisor making a mistake
        :param Q_state: Q-table entry to choose an action
        """
        self.name = name
        self.param = param
        self.Q_state = Q_state
        if name == 'guidance':
            if p_guidance_mistakes > 1 or p_guidance_mistakes < 0:
                raise ValueError("Probability of guidance error has to be value between 0 and 1")
            self.p_guidance_mistakes = p_guidance_mistakes

    def return_action(self):
        """
        Return new action according to the explroation strategy.
        :return: Action chosen to perform by an agent
        """
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
    def __init__(self, name, params):
        """
        Class representing an agents action-selection policy.
        :param name: type of the policy representation (e.g. qtable)
        :param params: any policy parameters (e.g. number of the available states and actions in the Q-table)
        """
        self.name = name
        self.params = params
        if self.name == 'qtable':
            [self.actlist, self.states] = self.params
            self.Q = [[0.0] * len(self.actlist) for x in range(len(self.states))]

class Learning:
    # qlearning, sarsa, traces, actor critic, policy gradient
    def __init__(self, name, params):
        """
        Class representing a reinforcement learning algorithm.
        :param name: type of the RL-algorithm (e.g. qlearn, sarsa)
        :param params: any paramteres needed for RL-algorithm (e.g. learning rate, discount factor)
        """
        self.name = name
        self.params = params
        if self.name == 'qlearn' or self.name == 'sarsa':
            self.alpha = self.params[0]
            self.gamma = self.params[1]

    def update(self, action, next_action, reward, Q_state, Q_next_state, done):
        """
        Update of a policy of the RL-algorithm
        :param action: current action to perform
        :param next_action: future action to perform (in the next state)
        :param reward: reward for the policy update
        :param Q_state: Q-table entry for the current state
        :param Q_next_state: Q-table entry for the future state
        :param done: flag indicating if the current session (10 sequences) is finished
        :return: updated Q-table entry, update error
        """
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
