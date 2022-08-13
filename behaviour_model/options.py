#!/usr/bin/python

'''
    Copyright 2022 by Michał Stolarz <michal.stolarz@h-brs.de>

    This file is part of migrave_personalised_behaviour_model,
    and is based on: https://github.com/TsiakasK/sequence-learning-dataset

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

import sys, getopt, os
from datetime import datetime
import numpy as np

ROOT_PATH = "behaviour_model"
def get_best_policy(table_name, epochs, runs_num):
    final_return_avgs = \
        np.array([np.loadtxt(os.path.join(ROOT_PATH, f"results/{table_name}/runs/{i}/return"),
                             dtype=float)[-epochs:].mean() for i in range(runs_num)])
    id_max = np.argmax(final_return_avgs)
    return id_max

def GetOptions(argv):
    user = 1
    pretrained_user = 1
    runs_num = 30 #30
    update_mode = 1  # mode=0 - performance score,
    # mode=1 - reward shaping (performance score + engagement),
    # mode=2 - engagement Q augmentation

    learning = 1
    exploration_policy = "exploitation" # "softmax", "egreedy", "exploitation"
    interactive_type = 0
    name = str(datetime.now())
    episodes = 40000
    epochs = 100
    reward_function = "square" # "normal", "double", "square"

    table_name = f"u{user}_m{update_mode}_policy-softmax_rewardfun-{reward_function}_pretrained-False"
    # Selection of the policy with the highest average return for the
    id_max = get_best_policy(table_name, epochs, runs_num)
    guidance_policy = os.path.join(ROOT_PATH, f"results/{table_name}/runs/{id_max}/q_table") #if there is learning from guidance
    p_guidance_mistakes = 0.2 #0.1 #0.2

    exploration = 300
    lr = 0.05 #0.05 for learning from guidance 1.0
    gamma = 0.95
    beta1 = 3
    beta2 = 3
    OKGREEN = '\033[92m'
    ENDC = '\033[0m'

    table_name = f"u{pretrained_user}_m{update_mode}_policy-softmax_rewardfun-{reward_function}_pretrained-False"
    # table_name = f"u{pretrained_user}_m{update_mode}_policy-softmax_pretrained-False"
    # table_name = f"u{user}_m{update_mode}_guidance_error-0.0_pretrained-False" # for guidance
    # Selection of the policy with the highest average return for the
    #id_max = get_best_policy(table_name, epochs, runs_num)
    Table = "" #os.path.join(ROOT_PATH, f"results/{table_name}/runs/{id_max}/q_table")

    if guidance_policy:
        name += f"_u{user}_m{update_mode}_guidance_error-{p_guidance_mistakes}_rewardfun-{reward_function}_pretrained-{Table.split('/')[-4] if Table else False}"
    else:
        name += f"_u{user}_m{update_mode}_policy-{exploration_policy}_rewardfun-{reward_function}_pretrained-{Table.split('/')[-4] if Table else False}"

    try:
        opts, args = getopt.getopt(argv, "he:q:p:l:u:n:i:t:a:")

    except getopt.GetoptError:
        print('\n' + OKGREEN + 'USAGE:\n')
        print(
            './sequence_learning.py -e episodes -p epochs -q qtable -u user -n name -l learning -i interactive_type '
            '-t exploration -a learning_rate' + ENDC + '\n')
        sys.exit(2)
    for opt, arg in opts:
        # print opt, arg
        if opt == '-h':
            print('\n' + OKGREEN + 'USAGE:\n')
            print(
                './sequence_learning.py -e episodes -p epochs -q qtable -u user -n name -l learning -i '
                'interactive_type\n')
            print("episodes in sumber of learning episodes (integer) -- default 5000")
            print("epochs is the number of episodes per epoch -- default 50")
            print("qtable is the name of the q_table file to load -- default is based on 'empty'")
            print("name is the name of the folder -- default is based on date")
            print("user is the user cluster (user1, user2) used for the experiment -- default 1")
            print("interactive_type is the selection of none (0), feedback (1), guidance (2), or both (3) -- default 0")
            print("learning: 0 for no learning and 1 for learning (Q-values update)-- default 1 \n\n" + ENDC)
            print(
                "update_mode is the selection mode=0 - performance score, mode=1 - reward shaping (performance score + engagement), mode=2 - engagement Q augmentation")
            print("exploration_policy - policy for solving the exploration/exploitation problem")
            print("runs_num - number of runs of the algorithm")
            sys.exit()
        elif opt in ("-q", "--qtable"):
            Table = arg
        elif opt in ("-t", "--exploration"):
            exploration = float(arg)
        elif opt in ("-e", "--episodes"):
            episodes = int(arg)
        elif opt in ("-u", "--user"):
            user = int(arg)
        elif opt in ("-p", "--epochs"):
            epochs = float(arg)
        elif opt in ("-n", "--name"):
            name = str(arg)
        elif opt in ("-i", "--interactive"):
            interactive_type = int(arg)
        elif opt in ("-l", "--learning"):
            learning = int(arg)
        elif opt in ("-a", "--alpha"):
            lr = float(arg)
        elif opt in ("-l", "--learning"):
            learning = int(arg)
        elif opt in ("-m", "--update_mode"):
            update_mode = int(arg)
        elif opt in ("-ep", "--exploration_policy"):
            exploration_policy = str(arg)
        elif opt in ("-g", "--guidance"):
            guidance = bool(arg)
        elif opt in ("-rn", "--runs_num"):
            runs_num = int(arg)

    if len(argv[1::]) == 0:
        print('\n' + OKGREEN + 'Running with default parameters...' + ENDC + '\n')

    return episodes, epochs, user, Table, name, learning, interactive_type, exploration, \
           lr, gamma, update_mode, beta1, beta2, exploration_policy, guidance_policy, runs_num, \
           p_guidance_mistakes, reward_function
