#!/usr/bin/python

'''
    Copyright 2022 by Michał Stolarz <michal.stolarz@h-brs.de>

    This file is part of migrave_personalised_behaviour_model.

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

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import os
import json
import itertools

matplotlib.rc('font', **{'size': 20})
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

def load_model(path):
    with open(path) as model_json:
        model_dict = json.load(model_json)

    new_model_dict = {}
    for key in model_dict:
        new_key = key.replace("(", "")
        new_key = new_key.replace(")", "")
        tuple_key = tuple([float(item) for item in new_key.split(',')])
        new_model_dict[tuple_key] = model_dict[key]

    return new_model_dict

MODEL_DIRECTORY = "../model"
MODEL_NAME = "gp"
USERS_ID = [0, 1]

length = [1/3, 2/3, 1.0]
feedback = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
previous = [-1.0, -2/3, -1/3, 0.0, 1/3, 2/3, 1.0]
combs = (length, feedback, previous)
states = list(itertools.product(*combs))
states = tuple([tuple([state[0], state[1][0], state[1][1], state[1][2], state[2]]) for state in states])

fig, axes = plt.subplots(len(USERS_ID), sharex=True, figsize=(10, 6), dpi=100)

for user_id, ax in zip(USERS_ID, axes):
    preds_succ = []
    preds_failure = []
    args_failure = []
    args_succ = []
    stds_succ = []
    stds_failure = []

    training_data = load_model(os.path.join(MODEL_DIRECTORY, f"user{user_id}_engagement_training.json"))
    predict_data_std = load_model(os.path.join(MODEL_DIRECTORY, f"user{user_id}_engagement_std.json"))
    predict_data = load_model(os.path.join(MODEL_DIRECTORY, f"user{user_id}_engagement.json"))

    state_level = [0, 21, 42]
    state_level_nums = [21, 21, 21]

    for result in [-1.0, 1.0]:
        for i, s in enumerate(states):

            pred = predict_data[tuple([s[0], s[1], s[2], s[3], s[4], result])]

            if MODEL_NAME == "gp":
                std = predict_data_std[tuple([s[0], s[1], s[2], s[3], s[4], result])]
                if result == 1:
                    stds_succ.append(float(std))
                elif result == -1:
                    stds_failure.append(float(std))

            if result == 1:
                preds_succ.append(float(pred))
                args_succ.append(i)
            elif result == -1:
                preds_failure.append(float(pred))
                args_failure.append(i)

    plotting_features = [("$E$ (failure)", 'red', args_failure, preds_failure, stds_failure, -1),
                         ("$E$ (success)", 'blue', args_succ, preds_succ, stds_succ, 1)]
    vmax = []

    for label, color, args, preds, stds, result in plotting_features:
        ax.plot(args, preds, label=f"predicted {label}", color=color)

        if MODEL_NAME == "gp":
            preds_ = np.array(preds)
            stds_ = np.array(stds)

            pos_shift_preds = np.array([value if value <= 1 else 1 for value in list(preds_ + stds_)])
            neg_shift_preds = np.array([value if value >= -1 else -1 for value in list(preds_ - stds_)])

            ax.fill_between(np.array(args),
                             y1=pos_shift_preds,
                             y2=neg_shift_preds,
                             color=f"tab:{color}",
                             alpha=0.2)
        state_x = []
        state_y = []
        for a in training_data:
            if a[5] == result:
                state_x.append(states.index(tuple([a[0], a[1], a[2], a[3], a[4]])))
                state_y.append(training_data[a])

        ax.plot(state_x, state_y, f"o{color[0]}", label=f"estimated {label}")

    for bound in [-1, 1]:
        ax.bar(state_level,
                height=[bound * 1.2] * len(state_level),
                width=state_level_nums,
                color=['w', 'k'] * int(len(state_level) / 2),
                align="edge",
                alpha=0.1)

    if user_id == 1:
        for i in range(len(state_level)):
            ax.text((state_level[i] + state_level_nums[i] / 2 - 3), 0.77, f"Level {i + 1}",
                    bbox={'facecolor': 'white', 'alpha': 0.3, 'edgecolor': 'black', 'pad': 3})

    ax.set_title(f"User Cluster {user_id + 1}")
    ax.grid()
    ax.set(ylim=(-1, 1), xlim=(0, len(states)-1))

plt.legend(bbox_to_anchor=(0.25, -1.3), loc='lower left')
fig.supxlabel("State id")
fig.supylabel("Engagement")
plt.savefig(f"engagement_{MODEL_NAME}.pdf", bbox_inches="tight")
plt.close()