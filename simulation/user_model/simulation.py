#!/usr/bin/python
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
