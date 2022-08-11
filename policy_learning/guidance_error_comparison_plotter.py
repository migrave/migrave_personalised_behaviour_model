#!/usr/bin/python

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


def average_data(data):
    tmp = []
    data_epoch = []
    for i, value in enumerate(data):
        tmp.append(value)
        if i % epochs == 0:
            a = np.asarray(tmp)
            data_epoch.append(a.mean())
            tmp = []
    return data_epoch

def calculate_limits(down_limit, up_limit, data_avg_epoch, data_std_epoch):
    up_shift = np.array(
        [value if value <= up_limit else up_limit for value in list(data_avg_epoch + data_std_epoch)])
    down_shift = np.array(
        [value if value >= down_limit else down_limit for value in list(data_avg_epoch - data_std_epoch)])
    return down_shift, up_shift


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

            data_avg_epoch = np.loadtxt(os.path.join(DATA_ROOT, DATA_DIRECTS[key], 'merged', f"{name}_avg"),
                                        dtype=float)
            data_std_epoch = np.loadtxt(os.path.join(DATA_ROOT, DATA_DIRECTS[key], 'merged', f"{name}_std"),
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
