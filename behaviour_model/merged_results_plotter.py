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
from behaviour_model.utils import average_data

epochs = 100

font = {'size': 20}
matplotlib.rc('font', **font)
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

ROOT_PATH = "behaviour_model"
DATA_ROOT = "results"
USER = 0
OUTPUT_DIR = "merged"

# For trained from scratch
DATA_DIRECT = "u0_m1_guidance_error-0.0_rewardfun-square_pretrained-False"

if not os.path.exists(f"results/{DATA_DIRECT}/{OUTPUT_DIR}"):
    os.makedirs(f"results/{DATA_DIRECT}/{OUTPUT_DIR}")

SC_raw = []
RET_raw = []
MS_raw = []
V_raw = []
ENG_raw = []
ERR_raw = []
CORR_raw = []

for run in os.listdir(os.path.join('results', DATA_DIRECT, 'runs')):
    SC_raw.append(np.loadtxt(f"results/{DATA_DIRECT}/runs/{run}/score", dtype=float))
    RET_raw.append(np.loadtxt(f"results/{DATA_DIRECT}/runs/{run}/return", dtype=float))
    MS_raw.append(np.loadtxt(f"results/{DATA_DIRECT}/runs/{run}/max_score", dtype=float))
    V_raw.append(np.loadtxt(f"results/{DATA_DIRECT}/runs/{run}/v_start", dtype=float))
    ENG_raw.append(np.loadtxt(f"results/{DATA_DIRECT}/runs/{run}/engagement", dtype=float))
    ERR_raw.append(np.loadtxt(f"results/{DATA_DIRECT}/runs/{run}/error", dtype=float))
    CORR_raw.append(np.loadtxt(f"results/{DATA_DIRECT}/runs/{run}/corrections", dtype=float))

SC = average_data(SC_raw)
sc_mean = np.mean(SC, axis=0)
np.savetxt(f"results/{DATA_DIRECT}/{OUTPUT_DIR}/score_avg", sc_mean, delimiter='\n')
sc_std = np.std(SC, axis=0)
np.savetxt(f"results/{DATA_DIRECT}/{OUTPUT_DIR}/score_std", sc_std, delimiter='\n')

RET = average_data(RET_raw)
ret_mean = np.mean(RET, axis=0)
np.savetxt(f"results/{DATA_DIRECT}/{OUTPUT_DIR}/return_avg", ret_mean, delimiter='\n')
ret_std = np.std(np.array(RET), axis=0)
np.savetxt(f"results/{DATA_DIRECT}/{OUTPUT_DIR}/return_std", ret_std, delimiter='\n')

V = average_data(V_raw)
v_mean = np.mean(V, axis=0)
np.savetxt(f"results/{DATA_DIRECT}/{OUTPUT_DIR}/v_start_avg", v_mean, delimiter='\n')
v_std = np.std(V, axis=0)
np.savetxt(f"results/{DATA_DIRECT}/{OUTPUT_DIR}/v_start_std", v_std, delimiter='\n')

ENG = average_data(ENG_raw)
eng_mean = np.mean(ENG, axis=0)
np.savetxt(f"results/{DATA_DIRECT}/{OUTPUT_DIR}/engagement_avg", eng_mean, delimiter='\n')
eng_std = np.std(ENG, axis=0)
np.savetxt(f"results/{DATA_DIRECT}/{OUTPUT_DIR}/engagement_std", eng_std, delimiter='\n')

ERR = average_data(ERR_raw)
err_mean = np.mean(ERR, axis=0)
np.savetxt(f"results/{DATA_DIRECT}/{OUTPUT_DIR}/error_avg", err_mean, delimiter='\n')
err_std = np.std(ERR, axis=0)
np.savetxt(f"results/{DATA_DIRECT}/{OUTPUT_DIR}/error_std", err_std, delimiter='\n')

CORR_epoch = average_data(CORR_raw)
corr_epoch_mean = np.mean(CORR_epoch, axis=0)
np.savetxt(f"results/{DATA_DIRECT}/{OUTPUT_DIR}/corrections_avg", corr_epoch_mean, delimiter='\n')
corr_epoch_std = np.std(CORR_epoch, axis=0)
np.savetxt(f"results/{DATA_DIRECT}/{OUTPUT_DIR}/corrections_std", corr_epoch_std, delimiter='\n')

CORR = CORR_raw
corr_mean = np.mean(CORR, axis=0)
np.savetxt(f"results/{DATA_DIRECT}/{OUTPUT_DIR}/corrections_all_avg", corr_mean, delimiter='\n')
corr_std = np.std(CORR, axis=0)
np.savetxt(f"results/{DATA_DIRECT}/{OUTPUT_DIR}/corrections_all_std", corr_std, delimiter='\n')

figure(figsize=(10, 6), dpi=400)

plt.plot(ret_mean)
plt.fill_between(np.array(range(ret_std.shape[0])),
                 y1=ret_mean - ret_std,
                 y2=ret_mean + ret_std,
                 color="tab:blue",
                 alpha=0.2)
plt.grid()
plt.xlabel("Epochs")
plt.ylabel("Return")
plt.savefig(f"results/{DATA_DIRECT}/{OUTPUT_DIR}/return.png")
plt.close()

figure(figsize=(10, 6), dpi=400)

plt.plot(eng_mean)
plt.fill_between(np.array(range(eng_mean.shape[0])),
                 y1=eng_mean - eng_std,
                 y2=eng_mean + eng_std,
                 color="tab:blue",
                 alpha=0.2)
plt.grid()
plt.xlabel("Epochs")
plt.ylabel("Engagement")
plt.savefig(f"results/{DATA_DIRECT}/{OUTPUT_DIR}/engagement.png")
plt.close()

figure(figsize=(10, 6), dpi=400)

plt.plot(v_mean)
plt.fill_between(np.array(range(v_mean.shape[0])),
                 y1=v_mean - v_std,
                 y2=v_mean + v_std,
                 color="tab:blue",
                 alpha=0.2)
plt.grid()
plt.xlabel("Epochs")
plt.ylabel("Mean V(s)")
plt.savefig(f"results/{DATA_DIRECT}/{OUTPUT_DIR}/mean_v(s).png")
plt.close()

figure(figsize=(10, 6), dpi=400)

plt.plot(sc_mean)
plt.fill_between(np.array(range(sc_mean.shape[0])),
                 y1=sc_mean - sc_std,
                 y2=sc_mean + sc_std,
                 color="tab:blue",
                 alpha=0.2)
plt.grid()
plt.xlabel("Epochs")
plt.ylabel("User acc points")
plt.savefig(f"results/{DATA_DIRECT}/{OUTPUT_DIR}/score.png")
plt.close()

figure(figsize=(10, 6), dpi=400)

plt.plot(err_mean)
plt.fill_between(np.array(range(err_mean.shape[0])),
                 y1=err_mean - err_std,
                 y2=err_mean + err_std,
                 color="tab:blue",
                 alpha=0.2)
plt.grid()
plt.xlabel("Epochs")
plt.ylabel("Error update")
plt.savefig(f"results/{DATA_DIRECT}/{OUTPUT_DIR}/error.png")
plt.close()

RATIO = average_data(np.array(SC_raw)/np.array(MS_raw))
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

# CORRECTIONS PLOTTING - HAS TO BE LIMITED, OTHERWISE LATEX ERROR
limit = 100

figure(figsize=(10, 6), dpi=400)
plt.plot(corr_mean[:limit])
plt.fill_between(np.array(range(corr_mean[:limit].shape[0])),
                 y1=corr_mean[:limit] - corr_std[:limit],
                 y2=corr_mean[:limit] + corr_std[:limit],
                 color="tab:blue",
                 alpha=0.2)
plt.grid()
plt.xlabel("Game session")
plt.ylabel("Corrections")
plt.savefig(f"results/{DATA_DIRECT}/{OUTPUT_DIR}/corrections.png")
plt.close()