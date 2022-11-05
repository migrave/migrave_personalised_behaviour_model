#!/usr/bin/python

'''
    Copyright 2022 by Michał Stolarz <michal.stolarz@h-brs.de>

    This file is part of migrave_personalised_behaviour_model,
    and is based on: https://github.com/TsiakasK/sequence-learning-dataset.
    It contains parameters neccessary for the behaviour model training.

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

import os
from datetime import datetime
import numpy as np
from behaviour_model.utils import get_best_policy

ROOT_PATH = "behaviour_model"
USER = 0 # Cluster ID, 0 and 1 are available
PRETRAINED_USER = 1 # ID of the user cluster which was used to train the policy used for policy initialisation
                    # (policy pretraining), 0 and 1 are available
RUNS_NUMBER = 30 # How many times the training should be run
UPDATE_MODE = 1 # reward is calculated only with:
                # update_mode=0 - performance score,
                # update_mode=1 - performance score + engagement,
                # update_mode=2 - engagement
LEARNING = True # if Q-table should be updated
EXPLORATION_POLICY = "softmax" # Method for choosing actions i.e. exploration policy:
                                # "softmax", "egreedy", "exploitation" (no exploration)
EPISODES_NUMBER = 40000
EPOCHS_NUMBER = 100
RE_FUNCTION = "normal" # Method for calculating the performance score RE:
                        # "normal", "double", "square"
EXPLORATION_INIT_PARAM = 300 # Initial parameter for exploration,
                            # e.g. for softmax exploration it is initial temperature
LEARNING_RATE = 0.05
DISCOUNT_FACTOR = 0.95
BETA1 = 3 # For shaping the influence of engagement in the reward calculation (performance score + beta1*engagement)
BETA2 = 3 # For shaping the influence of engagement in the reward calculation (beta2*engagement)

GUIDANCE = False # If learning from guidance is used
P_GUIDANCE_MISTAKES = 0.2 # Probability of the supervisor to make a mistake

PRETRAINED_POLICY = False # If pretrained policy is used

def get_params():
    global EXPLORATION_POLICY
    name = str(datetime.now())

    if GUIDANCE:
        guidance_table_name = f"u{USER}_m{UPDATE_MODE}_policy-softmax_rewardfun-{RE_FUNCTION}_pretrained-False"
        # Selection of the policy with the highest average return for the
        id_max = get_best_policy(guidance_table_name, ROOT_PATH, EPOCHS_NUMBER, RUNS_NUMBER)
        guidance_policy = os.path.join(ROOT_PATH, f"results/{guidance_table_name}/runs/{id_max}/q_table")
        EXPLORATION_POLICY = "exploitation"
    else:
        guidance_policy = ""

    if PRETRAINED_POLICY:
        pretrained_table_name = f"u{PRETRAINED_USER}_m{UPDATE_MODE}_policy-softmax_rewardfun-{RE_FUNCTION}_pretrained-False"
        # Selection of the policy with the highest average return for the
        id_max = get_best_policy(pretrained_table_name, ROOT_PATH, EPOCHS_NUMBER, RUNS_NUMBER)
        pretrained_policy = os.path.join(ROOT_PATH, f"results/{pretrained_table_name}/runs/{id_max}/q_table")
        EXPLORATION_POLICY = "exploitation"
    else:
        pretrained_policy = ""

    if guidance_policy:
        name += f"_u{USER}_m{UPDATE_MODE}_guidance_error-{P_GUIDANCE_MISTAKES}_rewardfun-{RE_FUNCTION}_pretrained-{pretrained_policy.split('/')[-4] if pretrained_policy else False}"
    else:
        name += f"_u{USER}_m{UPDATE_MODE}_policy-{EXPLORATION_POLICY}_rewardfun-{RE_FUNCTION}_pretrained-{pretrained_policy.split('/')[-4] if pretrained_policy else False}"

    return [EPISODES_NUMBER, EPOCHS_NUMBER, USER, pretrained_policy, name, LEARNING, EXPLORATION_INIT_PARAM, \
           LEARNING_RATE, DISCOUNT_FACTOR, UPDATE_MODE, BETA1, BETA2, EXPLORATION_POLICY, guidance_policy, RUNS_NUMBER, \
           P_GUIDANCE_MISTAKES, RE_FUNCTION]
