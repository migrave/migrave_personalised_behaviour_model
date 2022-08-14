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
import sys
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import itertools
sys.path.append('../..')
from behaviour_model.utils import get_best_policy

font = {'size': 8}
matplotlib.rc('font', **font)

matplotlib.rcParams.update({
    'font.family': 'serif',
    'pgf.rcfonts': False,
})

ROOT_PATH = ".."
reward_function = "square"
OUTPUT_DIR = "policy_visualization_guidance_error"
DATA_DIRECTS = {0: {0: f"u0_m1_policy-softmax_rewardfun-{reward_function}_pretrained-False",
                    1: f"u0_m1_policy-exploitation_rewardfun-{reward_function}_pretrained-u1_m1_policy-softmax_rewardfun-{reward_function}_pretrained-False",
                    2: f"u0_m1_guidance_error-0.0_rewardfun-{reward_function}_pretrained-False",
                    3: f"u0_m1_guidance_error-0.1_rewardfun-{reward_function}_pretrained-False",
                    4: f"u0_m1_guidance_error-0.2_rewardfun-{reward_function}_pretrained-False"},
                1: {0: f"u1_m1_policy-softmax_rewardfun-{reward_function}_pretrained-False",
                    1: f"u1_m1_policy-exploitation_rewardfun-{reward_function}_pretrained-u0_m1_policy-softmax_rewardfun-{reward_function}_pretrained-False",
                    2: f"u1_m1_guidance_error-0.0_rewardfun-{reward_function}_pretrained-False",
                    3: f"u1_m1_guidance_error-0.1_rewardfun-{reward_function}_pretrained-False",
                    4: f"u1_m1_guidance_error-0.2_rewardfun-{reward_function}_pretrained-False"}}

PLOTS_TITLES = {0: "Cold Start",
                1: "Pretrained",
                2: "G, $P(err)=0.0$",
                3: "G, $P(err)=0.1$",
                4: "G, $P(err)=0.2$"}


if __name__ == "__main__":
    runs_num = 30
    num_epochs_to_select_policy = 100 #number of episodes in one epoch

    length = [3, 5, 7]
    feedback = [0, 1, 2]
    previous = [-3, -2, -1, 0, 1, 2, 3]
    combs = (length, feedback, previous)

    for user_id in DATA_DIRECTS:
        fig, axes = plt.subplots(1, len(DATA_DIRECTS[user_id]), sharey=True, figsize=(7, 8), dpi=300)

        for id, ax in zip(DATA_DIRECTS[user_id], axes):
            table_name = DATA_DIRECTS[user_id][id]
            id_max = get_best_policy(table_name, ROOT_PATH, num_epochs_to_select_policy, runs_num)
            policy = os.path.join(ROOT_PATH, f"results/{table_name}/runs/{id_max}/q_table")  # if there is learning from guidance
            print(policy)

            with open(policy, 'r') as ins:
                Q = np.array([[float(n) for n in line.split()] for line in ins])

            actions = []
            states_ids = []
            ids_to_remove = np.where(~Q.any(axis=1))[0]

            for row_id, row in enumerate(Q):
                if row_id in ids_to_remove:
                    actions.append(-1)
                    states_ids.append(row_id)
                else:
                    indices = [index+1 for index, value in enumerate(row) if value == np.max(row)]
                    actions = [*actions, *indices]
                    states_ids = [*states_ids, *(len(indices)*[row_id])]

            #actions = 0
            states = list(itertools.product(*combs))
            states.append((0, 0, 0))
            states_str = [str(states[state_id]) for state_id in states_ids]
            y_pos = np.arange(len(states_str))

            ax.set_axisbelow(True)
            ax.set_title(PLOTS_TITLES[id])
            # ax.xaxis.grid()
            ax.grid(alpha=0.3, linewidth=0.8)
            ax.scatter(actions, states_ids, s=20, color="black")
            ax.set_yticks(states_ids, labels=states_str)
            ax.set_ylim(-1, y_pos[-1] + 1)
            ticks = [0, 1, 2, 3, 4, 5]
            ax.set_xticks(ticks)
            ax.set_xlim(0.6, len(ticks) - 0.6)
            ax.invert_yaxis()  # labels read top-to-bottom

        fig.supylabel("States")
        fig.supxlabel("Actions")
        plt.savefig(os.path.join(OUTPUT_DIR, f"user_{user_id}_rewardfun-{reward_function}.pdf"), bbox_inches='tight')
