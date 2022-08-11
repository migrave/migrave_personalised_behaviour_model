#!/usr/bin/python

import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np

font = {'size': 12}
matplotlib.rc('font', **font)
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})


USERS = [0, 1]
PRETRAINED_USERS = USERS[::-1]
reward_function = "square" #"normal", "double", "square"

DATA_ROOT = "results"
OUTPUT_DIR = "pretrain_comparison"

DATA_NAMES = ["score", "ratio", "engagement"]
AXIS_LABELS = {"score": "Performance Score",
               "ratio": "Success Ratio",
               "engagement": "Engagement"}

LEGEND = {0: r'$F_{r}=RE$',
          1: r'$F_{r}=RE+\beta E$',
          2: r'$F_{r}=\lambda E$'}

COLOR = {0: 'red',
         1: 'cyan',
         2: 'green'}

epochs = 100
episodes = 0

for name in DATA_NAMES:

    #figure(figsize=(10, 3), dpi=100)
    fig, axes = plt.subplots(len(USERS), sharex=True, figsize=(6, 5), dpi=100)

    for user, pre_user, ax in zip(USERS, PRETRAINED_USERS, axes):
        DATA_DIRECTS = {2: f"u{user}_m2_policy-softmax_pretrained-False",
                        1: f"u{user}_m1_policy-softmax_rewardfun-{reward_function}_pretrained-False",
                        0: f"u{user}_m0_policy-softmax_rewardfun-{reward_function}_pretrained-False"}
        PRETRAINED_DATA_DIRECT = f"u{user}_m1_policy-exploitation_rewardfun-{reward_function}_pretrained-u{pre_user}_m1_policy-softmax_rewardfun-{reward_function}_pretrained-False"

        # Plotting results from different update methods
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

        pre_data_avg_epoch = np.loadtxt(os.path.join(DATA_ROOT, PRETRAINED_DATA_DIRECT, 'merged', f"{name}_avg"),
                                        dtype=float)
        pre_data_std_epoch = np.loadtxt(os.path.join(DATA_ROOT, PRETRAINED_DATA_DIRECT, 'merged', f"{name}_std"),
                                        dtype=float)

        ax.plot(pre_data_avg_epoch, label='pretrained', color='blue')

        if name == 'ratio':
            up_shift_pre_ratio = np.array(
                [value if value <= 1 else 1 for value in list(pre_data_avg_epoch + pre_data_std_epoch)])
            down_shift_pre_ratio = np.array(
                [value if value >= 0 else 0 for value in list(pre_data_avg_epoch - pre_data_std_epoch)])
        else:
            up_shift_pre_ratio = pre_data_avg_epoch + pre_data_std_epoch
            down_shift_pre_ratio = pre_data_avg_epoch - pre_data_std_epoch

        ax.fill_between(np.array(range(pre_data_avg_epoch.shape[0])),
                        y1=down_shift_pre_ratio,
                        y2=up_shift_pre_ratio,
                        color="tab:blue",
                        alpha=0.2)

        episodes = data_avg_epoch.shape[0]
        ax.grid()
        ax.set_title(f"User Cluster {user+1}")

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

    plt.savefig(os.path.join(OUTPUT_DIR, f"update_pretraining_comparison_{name}_rewardfun_{reward_function}.pdf"),
                bbox_inches='tight')
