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

import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np

matplotlib.rc('font', **{'size': 12})
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

ROOT_PATH = "behaviour_model"
DATA_ROOT = "results"
USERS = [0, 1]
PRETRAINED_USERS = USERS[::-1]

OUTPUT_DIR = "guidance_comparison"
UPDATE_MODE = 1

# For trained from scratch
# DATA_DIRECTS = {0: f"u{USER}_m{UPDATE_MODE}_policy-softmax_pretrained-False",
#                 1: f"u{USER}_m{UPDATE_MODE}_policy-exploitation_pretrained-u{PRETRAINED_USER}_m{UPDATE_MODE}_policy-softmax_pretrained-False",
#                 2: f"u{USER}_m{UPDATE_MODE}_guidance_pretrained-False",
#                 3: f"u{USER}_m{UPDATE_MODE}_guidance_pretrained-u{PRETRAINED_USER}_m{UPDATE_MODE}_policy-softmax_pretrained-False"}

DATA_NAMES = ["score", "ratio", "engagement"]
AXIS_LABELS = {"score": "Performance Score",
               "ratio": "Success Ratio",
               "engagement": "Engagement"}

COLOR = {0: 'red',
         1: 'cyan',
         2: 'green',
         3: 'blue'}

LEGEND = {0: "cold start",
          1: f"pretrained",
          2: "guidance",
          3: f"guidance and pretrained"}

epochs = 100


for name in DATA_NAMES:
    fig, axes = plt.subplots(len(USERS), sharex=True, figsize=(6, 5), dpi=100)

    for user, pre_user, ax in zip(USERS, PRETRAINED_USERS, axes):
        DATA_DIRECTS = {0: f"u{user}_m{UPDATE_MODE}_policy-softmax_pretrained-False",
                        1: f"u{user}_m{UPDATE_MODE}_policy-exploitation_pretrained-u{pre_user}_m{UPDATE_MODE}_policy-softmax_pretrained-False",
                        2: f"u{user}_m{UPDATE_MODE}_guidance_error-0.0_pretrained-False",
                        3: f"u{user}_m{UPDATE_MODE}_guidance_error-0.0_pretrained-u{pre_user}_m{UPDATE_MODE}_policy-softmax_pretrained-False"}

        for key in DATA_DIRECTS:
            data = []

            data_avg_epoch = np.loadtxt(os.path.join(DATA_ROOT, DATA_DIRECTS[key], 'merged', f"{name}_avg"), dtype=float)
            data_std_epoch = np.loadtxt(os.path.join(DATA_ROOT, DATA_DIRECTS[key], 'merged', f"{name}_std"), dtype=float)

            ax.plot(data_avg_epoch, label=LEGEND[key], color=COLOR[key])

            if name == 'ratio':
                up_shift_ratio = np.array([value if value <= 1 else 1 for value in list(data_avg_epoch + data_std_epoch)])
                down_shift_ratio = np.array([value if value >= 0 else 0 for value in list(data_avg_epoch - data_std_epoch)])

                ax.fill_between(np.array(range(data_avg_epoch.shape[0])),
                                y1=down_shift_ratio,
                                y2=up_shift_ratio,
                                color=f"tab:{COLOR[key]}",
                                alpha=0.2)
            else:
                ax.fill_between(np.array(range(data_avg_epoch.shape[0])),
                                y1=data_avg_epoch - data_std_epoch,
                                y2=data_avg_epoch + data_std_epoch,
                                color=f"tab:{COLOR[key]}",
                                alpha=0.2)

        episodes = data_avg_epoch.shape[0]
        ax.grid()
        ax.set_title(f"User Cluster {user + 1}")

        if name == "score":
            ax.set(ylim=(0, 21), xlim=(0, episodes))
        elif name == "engagement":
            if user == 0:
                ax.set(ylim=(0.45, 0.7), xlim=(0, episodes))
            if user == 1:
                ax.set(ylim=(-0.7, -0.3), xlim=(0, episodes))
        elif name == "ratio":
            ax.set(ylim=(0.2, 1.05), xlim=(0, episodes))

    fig.supxlabel("Epochs")
    fig.supylabel(AXIS_LABELS[name])
    plt.legend()

    plt.savefig(os.path.join(ROOT_PATH, OUTPUT_DIR, f"guidance_comparison_{name}.pdf"),
                bbox_inches='tight')
