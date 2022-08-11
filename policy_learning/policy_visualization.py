#!/usr/bin/python

import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import itertools

font = {'size': 8}
matplotlib.rc('font', **font)
# matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': True,
#     'pgf.rcfonts': False,
# })

matplotlib.rcParams.update({
    'font.family': 'serif',
    'pgf.rcfonts': False,
})

reward_function = "double" #"normal", "double", "square"
OUTPUT_DIR = "policy_visualization"
DATA_DIRECTS = {0: {0: f"u0_m0_policy-softmax_rewardfun-{reward_function}_pretrained-False",
                    1: f"u0_m1_policy-softmax_rewardfun-{reward_function}_pretrained-False",
                    2: f"u0_m2_policy-softmax_pretrained-False",
                    3: f"u0_m1_policy-exploitation_rewardfun-{reward_function}_pretrained-u1_m1_policy-softmax_rewardfun-{reward_function}_pretrained-False"},
                1: {0: f"u1_m0_policy-softmax_rewardfun-{reward_function}_pretrained-False",
                    1: f"u1_m1_policy-softmax_rewardfun-{reward_function}_pretrained-False",
                    2: f"u1_m2_policy-softmax_pretrained-False",
                    3: f"u1_m1_policy-exploitation_rewardfun-{reward_function}_pretrained-u0_m1_policy-softmax_rewardfun-{reward_function}_pretrained-False"}}

PLOTS_TITLES = {0: r'$F_{r}=RE$',
                1: r'$F_{r}=RE+\beta E$',
                2: r'$F_{r}=\lambda E$',
                3: r'pretrained'}

def get_best_policy(table_name, epochs, runs_num):
    final_return_avgs = \
        np.array([np.loadtxt(f"results/{table_name}/runs/{i}/return", dtype=float)[-epochs:].mean()
                  for i in range(runs_num)])
    id_max = np.argmax(final_return_avgs)
    return id_max


if __name__ == "__main__":
    runs_num = 30
    num_epochs_to_select_policy = 100

    length = [3, 5, 7]
    feedback = [0, 1, 2]
    previous = [-3, -2, -1, 0, 1, 2, 3]
    combs = (length, feedback, previous)

    for user_id in DATA_DIRECTS:
        fig, axes = plt.subplots(1, len(DATA_DIRECTS[user_id]), sharey=True, figsize=(7, 8), dpi=300)

        for id, ax in zip(DATA_DIRECTS[user_id], axes):
            table_name = DATA_DIRECTS[user_id][id]
            id_max = get_best_policy(table_name, num_epochs_to_select_policy, runs_num)

            policy = f"results/{table_name}/runs/{id_max}/q_table"  # if there is learning from guidance
            print(policy)
            with open(policy, 'r') as ins:
                Q = np.array([[float(n) for n in line.split()] for line in ins])

            actions = np.argmax(Q, axis=1)+1
            actions[np.where(~Q.any(axis=1))[0]] = 0

            states = list(itertools.product(*combs))
            states.append((0, 0, 0))
            states_str = [str(state) for state in states]
            y_pos = np.arange(len(states_str))

            ax.set_axisbelow(True)
            ax.set_title(PLOTS_TITLES[id])
            # ax.xaxis.grid()
            ax.grid(alpha=0.3, linewidth=0.8)
            ax.barh(y_pos, actions, align='center')
            ax.set_yticks(y_pos, labels=states_str)
            ax.set_ylim(-1, y_pos[-1] + 1)
            ticks = [0, 1, 2, 3, 4, 5]
            ax.set_xticks(ticks)
            ax.set_xlim(0, len(ticks) - 0.9)
            ax.invert_yaxis()  # labels read top-to-bottom

        # ax.set_title('How fast do you want to go today?')
        fig.supylabel("States")
        fig.supxlabel("Actions")
        # plt.savefig(os.path.join(OUTPUT_DIR, f"policy_u{user}_m{update_mode}_pretrained-{pretrained_user}_guidance-{guidance}.pdf"),
        #            bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_DIR, f"user_{user_id}_rewardfun_{reward_function}.pdf"), bbox_inches='tight')