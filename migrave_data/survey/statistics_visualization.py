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

OUTPUT_DIR = ""
DIFFICULTIES = [3, 5, 7]
DATA_NAMES = ['difficulty', 'performance', 'engagement']

DATA = {'difficulty': {3: {'easy': 18,
                           'just right': 1,
                           'difficult': 0},
                       5: {'easy': 2,
                           'just right': 17,
                           'difficult': 0},
                       7: {'easy': 0,
                           'just right': 2,
                           'difficult': 17}},
        'performance': {3: {'below average': 0,
                            'average': 7,
                            'above average': 12},
                        5: {'below average': 2,
                            'average': 9,
                            'above average': 8},
                        7:  {'below average': 6,
                            'average': 12,
                            'above average': 1}},
        'engagement': {3: {'not engaged': 0,
                           'neutral': 9,
                           'engaged': 10},
                       5: {'not engaged': 0,
                           'neutral': 4,
                           'engaged': 15},
                       7: {'not engaged': 3,
                           'neutral': 5,
                           'engaged': 11}}}

LABELS = {'difficulty': {'easy': 'easy',
                        'just right': 'just \n right',
                        'difficult': 'difficult'},
          'performance': {'below average': 'below \n average',
                          'average': 'average',
                          'above average': 'above \n average'},
          'engagement': {'not engaged': 'not \n engaged',
                         'neutral': 'neutral',
                         'engaged': 'engaged'}}

COLOR = {3: "green",
         5: "blue",
         7: "red"}

FILLING = {3: 'o',
           5: 'x',
           7: '+'}

iteration_to_label = 3


if __name__ == "__main__":
    fig, axes = plt.subplots(1, len(DATA), sharey=True, figsize=(7, 3), dpi=300)
    width = 0.4
    iteration_legend = 0
    handlers = []
    labels = []

    for label, ax in zip(DATA, axes):

        for diff, x_dist in zip(DATA[label], [-width, 0.0, width]):
            data = DATA[label][diff]

            keys = [LABELS[label][key] for key in data.keys()]
            values = [data[key] for key in data]
            x_pos = np.arange(len(keys))*2

            ax.set_axisbelow(True)
            ax.set_title(label.capitalize())
            # ax.xaxis.grid()
            ax.grid(alpha=0.3, linewidth=0.8)
            ax.bar(x_pos+x_dist, values, align='center', color=COLOR[diff], width=width, edgecolor="black",
                   linewidth=0.7)

            if iteration_legend > 1:
                bar_label = ax.bar(x_pos + x_dist, values, align='center', color=COLOR[diff], width=width, edgecolor="black",
                                   linewidth=0.7)
                handlers.append(bar_label)
                labels.append(f"Length {diff}")

            ax.set_xticks(x_pos, labels=keys)
            ax.set_xlim(-1, x_pos[-1] + 1)
            ticks = np.arange(20)
            ax.set_yticks(ticks)
            ax.set_ylim(0, len(ticks) - 0.9)

        iteration_legend = iteration_legend + 1
        if iteration_legend > 1:
            plt.legend(handlers, labels)

        #fig.suptitle(f'Difficulty level {diff}')
        fig.supylabel("Participants")
        #fig.supxlabel("Actions")
        # plt.savefig(os.path.join(OUTPUT_DIR, f"policy_u{user}_m{update_mode}_pretrained-{pretrained_user}_guidance-{guidance}.pdf"),
        #            bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_DIR, f"survey_level.pdf"), bbox_inches='tight')
