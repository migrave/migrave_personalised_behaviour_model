#!/usr/bin/python

'''
    Copyright 2022 by Michał Stolarz <michal.stolarz@h-brs.de>

    This file is part of migrave_personalised_behaviour_model,
    and is based on: https://github.com/TsiakasK/sequence-learning-dataset.
    It contains implementation of the neural networks used for modelling engagement and performance of the user clusters.

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

from keras.models import Sequential
from keras.layers.core import Dense

def PerformanceNN():
    model = Sequential()
    model.add(Dense(20, input_dim=5, activation='linear'))
    model.add(Dense(10, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model


def EngagementNN():
    model = Sequential()
    model.add(Dense(20, input_dim=6, activation='tanh'))
    model.add(Dense(10, activation='tanh'))
    model.add(Dense(1, activation='tanh'))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model
