#!/usr/bin/python

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
    training_data = load_model(os.path.join(MODEL_DIRECTORY, f"user{user_id}_performance_training.json"))
    predict_data_std = load_model(os.path.join(MODEL_DIRECTORY, f"user{user_id}_performance_std.json"))
    predict_data = load_model(os.path.join(MODEL_DIRECTORY, f"user{user_id}_performance.json"))

    state_level = [0, 21, 42]
    state_level_nums = [21, 21, 21]

    preds = []

    preds = [predict_data[key] for key in predict_data]
    if MODEL_NAME == "gp":
        stds = [predict_data_std[key] for key in predict_data_std]

    ax.bar(state_level,
            height=[1.2] * len(state_level),
            width=state_level_nums,
            color=['w', 'k'] * int(len(state_level) / 2),
            align="edge",
            alpha=0.1)

    ax.plot(preds, label='predicted value')

    if MODEL_NAME == "gp":
        preds = np.array(preds)
        stds = np.array(stds)
        pos_shift_preds = np.array([value if value <= 1 else 1 for value in list(preds + stds)])
        neg_shift_preds = np.array([value if value >= 0 else 0 for value in list(preds - stds)])
        ax.fill_between(np.array(range(len(preds))),
                         y1=neg_shift_preds,
                         y2=pos_shift_preds,
                         color="tab:blue",
                         alpha=0.2)

    first = 1
    for a in training_data:
        if first:
            ax.plot(states.index(a), training_data[a], 'or', label='real value')
            first = 0
        else:
            ax.plot(states.index(a), training_data[a], 'or')

    if user_id == 1:
        for i in range(len(state_level)):
            ax.text((state_level[i] + state_level_nums[i] / 2 - 3), 0.05, f"Level {i + 1}",
                    bbox={'facecolor': 'white', 'alpha': 0.3, 'edgecolor': 'black', 'pad': 3})

    ax.set_title(f"User Cluster {user_id + 1}")
    ax.grid()
    ax.set(ylim=(0, 1.1), xlim=(0, len(states)-1))

fig.supxlabel("State id")
fig.supylabel("$P($success$|\mathbf{s})$")
plt.legend(bbox_to_anchor=(0.3, -0.8), loc='lower left')
plt.savefig(f"performance_{MODEL_NAME}.pdf", bbox_inches="tight")
plt.close()