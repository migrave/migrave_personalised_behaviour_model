#!/usr/bin/python

'''
    Copyright 2022 by Michał Stolarz <michal.stolarz@h-brs.de>

    This file is part of migrave_personalised_behaviour_model.
    It is used for merging the results from all the training runs.

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
sys.path.append('..')
from behaviour_model.utils import average_data

# Set this variable to the directory with the data to merge
DATA_DIRECT = "u0_m0_policy-softmax_pretrained-False"


font = {'size': 20}
matplotlib.rc('font', **font)
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

EPOCHS = 100 #number of episodes in one epoch
ROOT_PATH = ""
DATA_ROOT = "results"
OUTPUT_DIR = "merged"

if not os.path.exists(os.path.join(DATA_ROOT, DATA_DIRECT, OUTPUT_DIR)):
    os.makedirs(os.path.join(DATA_ROOT, DATA_DIRECT, OUTPUT_DIR))

SC_raw = []
RET_raw = []
MS_raw = []
V_raw = []
ENG_raw = []
ERR_raw = []
CORR_raw = []

if_corrections = True

names_map = {'return': 'Return',
             'engagement': 'Engagement',
             'v_start': 'Mean V(s)',
             'score': 'User acc points',
             'error': 'Error update',
             'succes_ratio': 'Success ratio',
             'corrections': 'Corrections',
             'supervisor_mistakes': 'Mistakes'}

data_list = [SC_raw, RET_raw, V_raw, ENG_raw, ERR_raw, CORR_raw, CORR_raw]
files_name_list = ['score', 'return', 'v_start', 'engagement', 'error', 'corrections', 'corrections_all']
runs_num = 0

for run in os.listdir(os.path.join(DATA_ROOT, DATA_DIRECT, 'runs')):
    runs_num = runs_num + 1
    SC_raw.append(np.loadtxt(os.path.join(DATA_ROOT, DATA_DIRECT, f"runs/{run}/score"), dtype=float))
    RET_raw.append(np.loadtxt(os.path.join(DATA_ROOT, DATA_DIRECT, f"runs/{run}/return"), dtype=float))
    MS_raw.append(np.loadtxt(os.path.join(DATA_ROOT, DATA_DIRECT, f"runs/{run}/max_score"), dtype=float))
    V_raw.append(np.loadtxt(os.path.join(DATA_ROOT, DATA_DIRECT, f"runs/{run}/v_start"), dtype=float))
    ENG_raw.append(np.loadtxt(os.path.join(DATA_ROOT, DATA_DIRECT, f"runs/{run}/engagement"), dtype=float))
    ERR_raw.append(np.loadtxt(os.path.join(DATA_ROOT, DATA_DIRECT, f"runs/{run}/error"), dtype=float))
    try:
        CORR_raw.append(np.loadtxt(os.path.join(DATA_ROOT, DATA_DIRECT, f"runs/{run}/corrections"), dtype=float))
    except FileNotFoundError as err:
        print(err)
        if_corrections = False

if not if_corrections:
    print("Number of supervisor corrections will not be plotted.")

for data_raw, file_name in zip(data_list, files_name_list):
    if not if_corrections and file_name in ['corrections', 'corrections_all']:
        continue

    if file_name != 'corrections_all':
        data = average_data(data_raw, EPOCHS)
    else:
        data = data_raw

    data_mean = np.mean(data, axis=0)
    np.savetxt(os.path.join(DATA_ROOT, DATA_DIRECT, OUTPUT_DIR, f"{file_name}_avg"), data_mean, delimiter='\n')
    data_std = np.std(data, axis=0)
    np.savetxt(os.path.join(DATA_ROOT, DATA_DIRECT, OUTPUT_DIR, f"{file_name}_std"), data_std, delimiter='\n')

    if file_name == 'corrections':
        continue

    if file_name != 'corrections_all':
        figure(figsize=(10, 6), dpi=400)
        plt.plot(data_mean)
        plt.fill_between(np.array(range(data_std.shape[0])),
                         y1=data_mean - data_std,
                         y2=data_mean + data_std,
                         color="tab:blue",
                         alpha=0.2)
        plt.grid()
        plt.xlabel("Epochs")
        plt.ylabel(names_map[file_name])

        if file_name == 'v_start':
            plt.savefig(os.path.join(DATA_ROOT, DATA_DIRECT, OUTPUT_DIR, 'mean_v(s).png'))
        else:
            plt.savefig(os.path.join(DATA_ROOT, DATA_DIRECT, OUTPUT_DIR, f"{file_name}.png"))
        plt.close()

    else:
        # CORRECTIONS PLOTTING - HAS TO BE LIMITED, OTHERWISE LATEX ERROR
        limit = 100
        plt.plot(data_mean[:limit])
        plt.fill_between(np.array(range(data_mean[:limit].shape[0])),
                         y1=data_mean[:limit] - data_std[:limit],
                         y2=data_mean[:limit] + data_std[:limit],
                         color="tab:blue",
                         alpha=0.2)
        plt.grid()
        plt.xlabel("Game session")
        plt.ylabel("Corrections")
        plt.savefig(os.path.join(DATA_ROOT, DATA_DIRECT, OUTPUT_DIR, 'corrections.png'))
        plt.close()


RATIO = average_data(np.array(SC_raw)/np.array(MS_raw), EPOCHS)
ratio_mean = np.mean(RATIO, axis=0)
np.savetxt(f"results/{DATA_DIRECT}/{OUTPUT_DIR}/ratio_avg", ratio_mean, delimiter='\n')
ratio_std = np.std(RATIO, axis=0)
np.savetxt(f"results/{DATA_DIRECT}/{OUTPUT_DIR}/ratio_std", ratio_std, delimiter='\n')

figure(figsize=(10, 6), dpi=400)

plt.plot(ratio_mean)
pos_shift_ratio = np.array([value if value <= 1 else 1 for value in list(ratio_mean + ratio_std)])
neg_shift_ratio = np.array([value if value >= 0 else 0 for value in list(ratio_mean - ratio_std)])
plt.fill_between(np.array(range(ratio_mean.shape[0])),
                 y1=neg_shift_ratio,
                 y2=pos_shift_ratio,
                 color="tab:blue",
                 alpha=0.2)
plt.grid()
plt.xlabel("Epochs")
plt.ylabel("Success ratio")
plt.savefig(f"results/{DATA_DIRECT}/{OUTPUT_DIR}/succes_ratio.png")
plt.close()

print(f"Finished merging data from {runs_num} runs from the directory {DATA_DIRECT}/runs. \n The merged data is saved under {DATA_DIRECT}/merged.")