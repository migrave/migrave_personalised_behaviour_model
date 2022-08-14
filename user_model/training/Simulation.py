#!/usr/bin/python

'''
    Copyright 2022 by Michał Stolarz <michal.stolarz@h-brs.de>

    This file is part of migrave_personalised_behaviour_model.
    It contains base abstract class with common functionalities for calsses responsible for training performance and
    engagement user models.

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

from abc import ABC, abstractmethod
import itertools


class Simulation(ABC):
    def __init__(self, C):
        self.C0 = C.loc[C['cluster'] == 0]
        self.C1 = C.loc[C['cluster'] == 1]

        self.P0 = self.C0[['engagement', 'length', 'robot_feedback', 'previous_score', 'current_result']]
        self.P1 = self.C1[['engagement', 'length', 'robot_feedback', 'previous_score', 'current_result']]

        self.D = [3, 5, 7]
        self.S = [-3, -2, -1, 0, 1, 2, 3]
        self.L = [1/3, 2/3, 1.0]
        self.RF = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        self.PS = [-1.0, -2/3, -1/3, 0.0, 1/3, 2/3, 1.0]

        combs = (self.L, self.RF, self.PS)
        states = list(itertools.product(*combs))
        self.states = [(state[0], state[1][0], state[1][1], state[1][2], state[2]) for state in states]

        # For visualization of the difficulty levels purpose
        self.state_level_nums = []
        self.state_level = []

        tmp = 0
        counter = 0
        for i, s in enumerate(self.states):
            if tmp == s[0]:
                counter += 1
            if tmp < s[0]:
                self.state_level.append(i)
                tmp = s[0]
                if i != 0:
                    self.state_level_nums.append(counter)
                    counter = 0
                counter += 1
            if i == len(self.states)-1:
                self.state_level_nums.append(counter)

        self.models = []

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def eval(self):
        pass
