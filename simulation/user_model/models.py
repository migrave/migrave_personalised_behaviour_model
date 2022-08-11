#!/usr/bin/python

from keras.models import Sequential
from keras.layers.core import Dense

def PerformanceNN():
    model = Sequential()
    model.add(Dense(20, input_dim=5, activation='linear'))
    model.add(Dense(10, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model


def FeedbackNN():
    model = Sequential()
    model.add(Dense(20, input_dim=6, activation='tanh'))
    model.add(Dense(10, activation='tanh'))
    model.add(Dense(1, activation='tanh'))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model
