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

ROOT_PATH = ".."
DATA_ROOT = "results"
USERS = [0, 1]
PRETRAINED_USERS = USERS[::-1]

OUTPUT_DIR = "pretrain_comparison"

# For trained from scratch
# DATA_DIRECTS = {2: f"u{USER}_m2_policy-softmax_pretrained-False",
#                 1: f"u{USER}_m1_policy-softmax_pretrained-False",
#                 0: f"u{USER}_m0_policy-softmax_pretrained-False"}


# For user pretrained policy for
# PRETRAINED_DATA_DIRECTS = {
#     2: f"u{USER}_m2_policy-exploitation_pretrained-u{PRETRAINED_USER}_m2_policy-softmax_pretrained-False",
#     1: f"u{USER}_m1_policy-exploitation_pretrained-u{PRETRAINED_USER}_m1_policy-softmax_pretrained-False",
#     0: f"u{USER}_m0_policy-exploitation_pretrained-u{PRETRAINED_USER}_m0_policy-softmax_pretrained-False"}


DATA_NAMES = ["score", "ratio", "engagement"]
AXIS_LABELS = {"score": "Performance score",
               "ratio": "Success ratio",
               "engagement": "Engagement"}

LEGEND = {0: "performance score",
          1: "performance score+engagement",
          2: "engagement"}

COLOR = {0: 'red',
         1: 'cyan',
         2: 'green'}

epochs = 100 #number of episodes in one epoch


for name in DATA_NAMES:
    # figure(figsize=(10, 3), dpi=100)
    fig, axes = plt.subplots(len(USERS), sharex=True, figsize=(6, 5), dpi=100)

    for user, pre_user, ax in zip(USERS, PRETRAINED_USERS, axes):
        # For trained from scratch
        DATA_DIRECTS = {1: f"u{user}_m1_policy-softmax_pretrained-False"}
        PRETRAINED_DATA_DIRECTS = {
            1: f"u{user}_m1_policy-exploitation_pretrained-u{pre_user}_m1_policy-softmax_pretrained-False"}

        for key in DATA_DIRECTS:
            data = []
            data_avg_epoch = np.loadtxt(os.path.join(ROOT_PATH, DATA_ROOT, DATA_DIRECTS[key], 'merged', f"{name}_avg"),
                                        dtype=float)
            data_std_epoch = np.loadtxt(os.path.join(ROOT_PATH, DATA_ROOT, DATA_DIRECTS[key], 'merged', f"{name}_std"),
                                        dtype=float)

            pre_data_avg_epoch = np.loadtxt(os.path.join(ROOT_PATH, DATA_ROOT, PRETRAINED_DATA_DIRECTS[key], 'merged',
                                                         f"{name}_avg"), dtype=float)
            pre_data_std_epoch = np.loadtxt(os.path.join(ROOT_PATH, DATA_ROOT, PRETRAINED_DATA_DIRECTS[key], 'merged',
                                                         f"{name}_std"), dtype=float)

            ax.plot(data_avg_epoch, label="cold start", color='red')
            ax.plot(pre_data_avg_epoch, label=f"pretrained on user cluster {pre_user+1}", color='blue')

            if name == 'ratio':
                up_shift_ratio = np.array([value if value <= 1 else 1 for value in list(data_avg_epoch +
                                                                                        data_std_epoch)])
                down_shift_ratio = np.array([value if value >= 0 else 0 for value in list(data_avg_epoch -
                                                                                          data_std_epoch)])

                up_shift_pre_ratio = np.array(
                    [value if value <= 1 else 1 for value in list(pre_data_avg_epoch + pre_data_std_epoch)])
                down_shift_pre_ratio = np.array(
                    [value if value >= 0 else 0 for value in list(pre_data_avg_epoch - pre_data_std_epoch)])
            else:
                up_shift_ratio = data_avg_epoch + data_std_epoch
                down_shift_ratio = data_avg_epoch - data_std_epoch
                up_shift_pre_ratio = pre_data_avg_epoch + pre_data_std_epoch
                down_shift_pre_ratio = pre_data_avg_epoch - pre_data_std_epoch

            ax.fill_between(np.array(range(data_avg_epoch.shape[0])),
                             y1=down_shift_ratio,
                             y2=up_shift_ratio,
                             color="tab:red",
                             alpha=0.2)

            ax.fill_between(np.array(range(pre_data_avg_epoch.shape[0])),
                             y1=down_shift_pre_ratio,
                             y2=up_shift_pre_ratio,
                             color="tab:blue",
                             alpha=0.2)

        episodes = data_avg_epoch.shape[0]
        ax.grid()
        ax.set_title(f"User cluster {user + 1}")
        ax.legend()

        if name == "score":
            ax.set(ylim=(8, 21), xlim=(0, episodes))
        elif name == "engagement":
            if user == 0:
                ax.set(ylim=(0.45, 0.8), xlim=(0, episodes))
            if user == 1:
                ax.set(ylim=(-0.7, -0.2), xlim=(0, episodes))
        elif name == "ratio":
            ax.set(ylim=(0.4, 1.05), xlim=(0, episodes))

    fig.supxlabel("Epochs")
    fig.supylabel(AXIS_LABELS[name])

    plt.savefig(os.path.join(OUTPUT_DIR, f"pretrain_comparison_{name}.pdf"),
                bbox_inches='tight')
