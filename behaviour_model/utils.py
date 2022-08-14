#!/usr/bin/python

'''
    Copyright 2022 by Michał Stolarz <michal.stolarz@h-brs.de>

    This file is part of migrave_personalised_behaviour_model,
    and is based on: https://github.com/TsiakasK/sequence-learning-dataset

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
import os

def maxs(seq):
    """
    Return the list of position of the largest element, if there are several largest elements,
    chose randomly between them.
    :param seq: Sequence to find the maximum value
    :return: position of the largest element
    """
    max_indices = []
    max_val = seq[0]
    for i, val in ((i, val) for i, val in enumerate(seq) if val >= max_val):
        if val == max_val:
            max_indices.append(i)
        else:
            max_val = val
            max_indices = [i]
    return random.choice(max_indices)

def average_data(data, epochs):
    """
    Average the data over the specified number of epochs.
    :param data: data to average
    :param epochs: number of episodes in one epoch
    :return: averaged data
    """
    runs = []
    for row in data:
        tmp = []
        data_epoch = []
        for i, value in enumerate(row):
            tmp.append(value)
            if i % epochs == 0:
                a = np.asarray(tmp)
                data_epoch.append(a.mean())
                tmp = []
        runs.append(data_epoch)
    return runs

def calculate_limits(down_limit, up_limit, data_avg_epoch, data_std_epoch):
    """
    Crop data standard deviation envelope to the predefined limits.
    :param down_limit: bottom value limit
    :param up_limit: upper value limit
    :param data_avg_epoch: data averaged over the epochs
    :param data_std_epoch: standard deviation of the data averaged over the epochs
    :return: data with std envelope limited to the bottom and upper limit
    """
    up_shift = np.array(
        [value if value <= up_limit else up_limit for value in list(data_avg_epoch + data_std_epoch)])
    down_shift = np.array(
        [value if value >= down_limit else down_limit for value in list(data_avg_epoch - data_std_epoch)])
    return down_shift, up_shift

def get_best_policy(table_name, root_path, epochs, runs_num):
    """
    Selecting the best policy over the predefined number of training runs.
    Policy is chosen based on the highest average return value in the last training epoch.
    :param table_name: name of the directory to look for the policy
    :param root_path: name of the behaviour model main directory
    :param epochs: number of episodes in one epoch
    :param runs_num: number of runs from whihc the pbest policy should be selected
    :return: id of the best selected policy
    """
    final_return_avgs = \
        np.array([np.loadtxt(os.path.join(root_path, f"results/{table_name}/runs/{i}/return"),
                             dtype=float)[-epochs:].mean() for i in range(runs_num)])
    id_max = np.argmax(final_return_avgs)
    return id_max