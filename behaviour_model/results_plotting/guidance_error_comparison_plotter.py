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
from behaviour_model.utils import calculate_limits

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

OUTPUT_DIR = "guidance_comparison"
UPDATE_MODE = 1

DATA_NAMES = ["score", "ratio", "engagement", "corrections", "corrections_all"]
AXIS_LABELS = {"score": "Performance Score",
               "ratio": "Success Ratio",
               "engagement": "Engagement",
               "corrections": "Corrections",
               "corrections_all": "Corrections"}

COLOR = {0: 'red',
         1: 'cyan',
         2: 'green',
         3: 'blue',
         4: 'brown'}

LEGEND = {0: "cold start",
          1: r'G, $P(err)=0.0$',
          2: r'G, $P(err)=0.1$',
          3: r'G, $P(err)=0.2$'}

epochs = 100

mode = "square"


for name in DATA_NAMES:
    fig, axes = plt.subplots(len(USERS), sharex=True, figsize=(6, 5), dpi=100)

    for user, pre_user, ax in zip(USERS, PRETRAINED_USERS, axes):
        DATA_DIRECTS = {0: f"u{user}_m{UPDATE_MODE}_policy-softmax_rewardfun-{mode}_pretrained-False",
                        1: f"u{user}_m{UPDATE_MODE}_guidance_error-0.0_rewardfun-{mode}_pretrained-False",
                        2: f"u{user}_m{UPDATE_MODE}_guidance_error-0.1_rewardfun-{mode}_pretrained-False",
                        3: f"u{user}_m{UPDATE_MODE}_guidance_error-0.2_rewardfun-{mode}_pretrained-False"}

        for key in DATA_DIRECTS:
            if name in ['corrections', 'corrections_all'] and (key == 0 or 'guidance' not in DATA_DIRECTS[key]):
                continue

            data = []

            data_avg_epoch = np.loadtxt(os.path.join(ROOT_PATH, DATA_ROOT, DATA_DIRECTS[key], 'merged', f"{name}_avg"),
                                        dtype=float)
            data_std_epoch = np.loadtxt(os.path.join(ROOT_PATH, DATA_ROOT, DATA_DIRECTS[key], 'merged', f"{name}_std"),
                                        dtype=float)

            ax.plot(data_avg_epoch, label=LEGEND[key], color=COLOR[key])

            if name == 'ratio':
                down_shift, up_shift = calculate_limits(0, 1, data_avg_epoch, data_std_epoch)
            elif name in ['corrections', 'corrections_all']:
                down_shift, up_shift = calculate_limits(0, 10, data_avg_epoch, data_std_epoch)
            else:
                down_shift, up_shift = data_avg_epoch - data_std_epoch, data_avg_epoch + data_std_epoch

            ax.fill_between(np.array(range(data_avg_epoch.shape[0])),
                            y1=down_shift,
                            y2=up_shift,
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
        elif name == "corrections":
            ax.set(ylim=(0, 10.01), xlim=(0, episodes))
        elif name == "corrections_all":
            lim = 15
            ax.set(ylim=(0, 10.01), xlim=(0, lim), xticks=(np.arange(lim)), yticks=(np.arange(11)))

    fig.supxlabel("Epochs")

    if name == "corrections_all":
        fig.supxlabel("Game Sessions")

    fig.supylabel(AXIS_LABELS[name])
    plt.legend()

    plt.savefig(os.path.join(OUTPUT_DIR, f"guidance_error_comparison_{name}_mode_{mode}.pdf"),
                bbox_inches='tight')
