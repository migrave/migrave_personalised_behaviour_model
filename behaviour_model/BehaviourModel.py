#!/usr/bin/python

'''
    Copyright 2022 by Michał Stolarz <michal.stolarz@h-brs.de>

    This file is part of migrave_personalised_behaviour_model,
    and is based on: https://github.com/TsiakasK/sequence-learning-dataset.
    It contains the BehaviourModel class responsible for training the behaviour model and Logger class used for
    logging the training results.

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

import sys, os
import csv

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import random
import itertools
import random
from datetime import datetime
import json
from behaviour_model.rl_utils import MDP, Policy, Learning, Representation

class Logger:
    def __init__(self, log_path, run):
        self.log_path = log_path
        self.run = run
        self.run_log_path = os.path.join(log_path, f"runs/{run}")

        if not os.path.exists(self.run_log_path):
            os.makedirs(self.run_log_path)

        self.g = open(os.path.join(self.run_log_path, "episodes"), 'w')
        self.rr = open(os.path.join(self.run_log_path, "return"), 'w')
        self.ss = open(os.path.join(self.run_log_path, "score"), 'w')
        self.ms = open(os.path.join(self.run_log_path, "max_score"), 'w')
        self.vv = open(os.path.join(self.run_log_path, "v_start"), 'w')
        self.ee = open(os.path.join(self.run_log_path, "engagement"), 'w')
        self.er = open(os.path.join(self.run_log_path, "error"), 'w')
        self.corr = open(os.path.join(self.run_log_path, "corrections"), 'w')
        self.mist = open(os.path.join(self.run_log_path, "supervisor_mistakes"), 'w')

    def log_parameters(self, name, episodes, epochs, user, q_table, learn, alpha, gamma,
                       To, update_mode, exploration_policy, guidance_policy,
                       p_guidance_mistakes, reward_function, runs_num, beta1, beta2):

        with open(os.path.join(self.run_log_path, "logfile"), 'w') as logfile:
            logfile.write(f"Logfile for: {name} - {datetime.now()} \n\n")
            logfile.write(f"Number of episodes: {episodes} \n")
            logfile.write(f"Number of epochs: {epochs} \n")
            logfile.write(f"User ID: {user} \n")
            logfile.write(f"Qtable: {q_table} \n")
            logfile.write(f"If the agent should learn: {learn} \n")
            logfile.write(f"Learning Rate: {alpha} \n")
            logfile.write(f"Discount factor: {gamma} \n")
            logfile.write(f"Initial parameter for exploration: {To} \n")
            logfile.write(f"Update mode for reward shaping: {update_mode} \n")
            logfile.write(f"Exploration policy: {exploration_policy} \n")
            logfile.write(f"Guidance policy: {guidance_policy} \n")
            logfile.write(f"Probability of supervisor mistakes: {p_guidance_mistakes} \n")
            logfile.write(f"RE form (normal, double, square): {reward_function} \n")
            logfile.write(f"Numbers of runs: {runs_num} \n")
            logfile.write(f"beta1 (for game performance shaping): {beta1} \n")
            logfile.write(f"beta2 (for engagement shaping): {beta2} \n")

    def log_episodes(self, message):
        self.g.write(message)

    def log_return(self, message):
        self.rr.write(message)

    def log_score(self, message):
        self.ss.write(message)

    def log_max_score(self, message):
        self.ms.write(message)

    def log_value(self, message):
        self.vv.write(message)

    def log_engagement(self, message):
        self.ee.write(message)

    def log_error(self, message):
        self.er.write(message)

    def log_corrections(self, message):
        self.corr.write(message)

    def log_supervisor_mistakes(self, message):
        self.mist.write(message)

    def stop(self):
        self.vv.close()
        self.ss.close()
        self.ms.close()
        self.ee.close()
        self.er.close()
        self.g.close()
        self.rr.close()
        self.corr.close()
        self.mist.close()


class BehaviourModel:
    def __init__(self, performance_model, engagement_model, params):
        self.params = params
        self.root_path = "behaviour_model"
        self.user_model_path = "user_model/output/model"
        self.engagement_model = self.load_model(engagement_model)
        self.performance_model = self.load_model(performance_model)

    def load_model(self, path):
        """
        Load the user model
        :param path: path to the user model
        :return: loaded model in the form of dictionary
        """
        with open(path) as model_json:
            model_dict = json.load(model_json)

        new_model_dict = {}
        for key in model_dict:
            new_key = key.replace("(", "")
            new_key = new_key.replace(")", "")
            tuple_key = tuple([float(item) for item in new_key.split(',')])
            new_model_dict[tuple_key] = model_dict[key]

        return new_model_dict

    def state_action_space(self):
        """
        Create the state and action space
        :return: state space, normalised state space, action space
        """
        length = [3, 5, 7]
        feedback = [0, 1, 2]
        previous = [-3, -2, -1, 0, 1, 2, 3]
        combs = (length, feedback, previous)
        states = list(itertools.product(*combs))
        states.append((0, 0, 0)) # Start state where actions with the feedback can not be applied

        l = [1, 2, 3]
        f = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        combs = (l, f, previous)
        normalized_states = list(itertools.product(*combs))
        normalized_states.append((0, [0, 0, 0], 0)) # Start state where actions with the feedback can not be applied

        actions = [0, 1, 2, 3, 4]
        return states, normalized_states, actions

    def get_engagement(self, state, result):
        """
        Get engagement from the user model
        :param state: state for which the engagement should be obtained
        :param result: result of solving the sequence by the performance user model
        :return: engagement score
        """
        outcome = 1 if result > 0 else -1
        st = tuple([state[0] / 3.0, state[1][0], state[1][1], state[1][2], state[2] / 3.0, outcome])
        engagement = self.engagement_model[st]
        return engagement

    def get_next_state(self, state, states, normalized_states, action, previous):
        """
        Get next state
        :param state: state in which the agent currently is
        :param states: state space
        :param normalized_states: normalized state space
        :param action: action performed by the agent
        :param previous: result obtained by the user in the previous sequence
        :return: score (result of solving the sequence by the model), new state
        """
        levels = {3: 1, 5: 2, 7: 3}
        if action == 0:
            feedback = 0
            length = 3
        if action == 1:
            feedback = 0
            length = 5
        if action == 2:
            feedback = 0
            length = 7
        if action == 3:
            feedback = 1
            length = state[0]
        if action == 4:
            feedback = 2
            length = state[0]

        next_state = [length, feedback, previous]
        normalized_next_state = normalized_states[states.index(tuple(next_state))]
        st = tuple([normalized_next_state[0] / 3.0, normalized_next_state[1][0], normalized_next_state[1][1],
                    normalized_next_state[1][2], normalized_next_state[2] / 3.0])
        prob = self.performance_model[st]

        if random.random() <= prob:
            success = 1
        else:
            success = -1

        score = success * levels[length]
        return score, [length, feedback, previous]

    def save_plot(self, data, epochs, run, dir_name, data_name):
        """
        Generate and save a plot
        :param data: data to be plotted
        :param epochs: numer of training epochs
        :param run: ID of the training run
        :param dir_name: name of the directory to save the plot
        :param data_name: name of the data to be plotted
        :return: None
        """
        names_map = {'return': 'Return',
                     'engagement': 'Engagement',
                     'mean_v(s)': 'Mean V(s)',
                     'score': 'User acc points',
                     'error': 'Error update',
                     'succes_ratio': 'Success ratio',
                     'corrections': 'Corrections',
                     'supervisor_mistakes': 'Mistakes'}

        figure(figsize=(10, 6), dpi=400)
        tmp = []
        epoch_data = []

        if data_name != "succes_ratio":
            for i, t in enumerate(data):
                tmp.append(t)
                if i % epochs == 0:
                    a = np.asarray(tmp)
                    epoch_data.append(a.mean())
                    tmp = []
        else:
            for i, (maxs, s) in enumerate(data):
                tmp.append(s / maxs)
                if i % epochs == 0:
                    a = np.asarray(tmp)
                    epoch_data.append(a.mean())
                    tmp = []

        plt.plot(epoch_data)
        plt.grid()
        plt.xlabel("Epochs")
        plt.ylabel(names_map[data_name])
        plt.savefig(os.path.join(self.root_path, f"results/{dir_name}/runs/{run}/{data_name}.png"))
        plt.close()

    def save_qtable(self, Q, dir_name, run):
        """
        Save a Q-table obtained after training
        :param Q: Q-table
        :param dir_name: name of the directory to save the plot
        :param run: ID of the training run
        :return: None
        """
        with open(os.path.join(self.root_path, f"results/{dir_name}/runs/{run}/q_table"), 'w') as f:
            writer = csv.writer(f, delimiter=' ')
            writer.writerows(Q)

    def save_policy(self, states, exploration_policy, Q, dir_name, run):
        """
        Save RL policy obtained after training
        :param states: state space
        :param exploration_policy: Policy used for exploration (object of type Policy)
        :param Q: Q-table
        :param dir_name: name of the directory to save the plot
        :param run: ID of the training run
        :return: None
        """
        with open(os.path.join(self.root_path, f"results/{dir_name}/runs/{run}/policy"), 'w') as pf:
            for s, q in zip(states, Q):
                state_index = states.index(tuple(s))
                pf.write(str(state_index) + ' ' + str(s))

                for i in q:
                    softm = (np.exp(i / exploration_policy.param) / np.sum(np.exp(q / exploration_policy.param)))
                    pf.write(' ' + str(softm))
                pf.write('\n')

    def train(self):
        """
        Train the behaviour model
        :return: None
        """
        [episodes, epochs, user, q_table, name, learn, To, alpha, gamma, \
        update_mode, beta1, beta2, exploration_policy, guidance_policy, runs_num, \
        p_guidance_mistakes, reward_function] = self.params

        log_path = os.path.join(self.root_path , f"results/{name}")

        if not os.path.exists(log_path):
            os.makedirs(log_path)
            os.makedirs(os.path.join(log_path, "runs"))

        for run in range(runs_num):
            # Logging
            logger = Logger(log_path, run)
            logger.log_parameters(name, episodes, epochs, user, q_table, learn, alpha, gamma,
                                  To, update_mode, exploration_policy, guidance_policy,
                                  p_guidance_mistakes, reward_function, runs_num, beta1, beta2)

            # State space
            states, normed_states, actions = self.state_action_space()
            A = ['L = 3', 'L = 5', 'L = 7', 'PF', 'NF']
            first_length = random.choice([3, 5, 7])
            start_state = (0, 0, 0)
            start_state_index = states.index(tuple(start_state))
            m = MDP(start_state, actions)
            m.states = states

            # Qtable
            table = Representation('qtable', [m.actlist, m.states])
            pretrained = False
            Q_guidance = np.asarray(table.Q)
            Q = np.asarray(table.Q)

            # Policies
            exp_strategy = Policy(name=exploration_policy, param=To)
            if guidance_policy:
                print('Loading Q-table guidance policy: ' + str(guidance_policy))
                with open(guidance_policy, 'r') as ins:
                    Q_guidance = np.array([[float(n) for n in line.split()] for line in ins])
                exp_strategy = Policy(name="exploitation", param=To)
                guidance_exp_strategy = Policy(name="guidance", param=To, p_guidance_mistakes=p_guidance_mistakes)
            if q_table:
                print('Loading Q-table: ' + str(q_table))
                pretrained = True
                with open(q_table, 'r') as ins:
                    Q = np.array([[float(n) for n in line.split()] for line in ins])

            # Initialisation
            table.Q = Q
            learning = Learning('qlearn', [alpha, gamma])

            R = []
            V = []
            S = []
            ENG = []
            CORRECTIONS = []
            SUPERVISOR_MISTAKES = []
            ER = []
            MS = []
            print(start_state_index)
            visits = np.ones((len(states) + 1))
            episode = 1
            first_reward = 1
            score_map = {3: 1, 5: 2, 7: 3}

            # Learning loop
            while (episode < episodes):
                state_index = start_state_index
                state = start_state
                score = 0
                max_score = 0
                iteration = 1
                end_game = 0
                done = 0
                r = 0
                quit_signal = 0
                N = 10
                previous_result = 0
                corrections = 0
                supervisor_mistakes = 0
                EE = []
                ERROR = []
                random.seed(datetime.now())

                if episode % epochs == 0 or episode == 1:
                    logger.log_episodes('Episode No.' + str(episode) + '\n')
                    print('Episode No.' + str(episode) + '\n')

                while (not done):
                    state_index = states.index(tuple(state))
                    exp_strategy.Q_state = Q[state_index][:]

                    if guidance_policy:
                        guidance_exp_strategy.Q_state = Q_guidance[state_index][:]

                    # adaptive exploration per state visit
                    exp_strategy.param = To - 5 * float(visits[state_index])

                    if exp_strategy.param < 0.5:
                        exp_strategy.param = 0.5

                    if episode % epochs == 0:
                        visits[state_index] += 1

                    # robot feedback (actions 3,4) is not available in the first state
                    if state_index == start_state_index:
                        exp_strategy.Q_state = Q[state_index][:3]
                        if guidance_policy:
                            guidance_exp_strategy.Q_state = Q_guidance[state_index][:3]

                    action = exp_strategy.return_action()

                    result, next_state = self.get_next_state(state, states, normed_states, action, previous_result)
                    next_state_index = states.index(tuple(next_state))

                    if reward_function == "normal":
                        reward = result if result > 0.0 else -1.0
                    elif reward_function == "double":
                        reward = result*2 if result > 0.0 else -1.0
                    elif reward_function == "square":
                        reward = result**2 if result > 0.0 else -1.0
                    else:
                        raise NotImplementedError

                    score += result
                    max_score += score_map[next_state[0]]
                    engagement = self.get_engagement(normed_states[next_state_index], result)
                    EE.append(engagement)

                    if update_mode == 1:
                        reward += beta1 * engagement
                    elif update_mode == 2:
                        reward = beta2 * engagement

                    r += (learning.gamma ** (iteration - 1)) * reward

                    next_action = 0 #Because Q-learning is used

                    if episode % epochs == 0 or episode == 1:
                        logger.log_episodes(str(iteration) + '... ' + str(state) + ' ' + str(A[action]) +
                                ' ' + str(next_state) + ' ' + str(reward) + ' ' + str(score) +
                                ' ' + str(engagement) + '\n')

                    if iteration == N:
                        done = 1

                    iteration += 1

                    error = 0
                    if learn:
                        if guidance_policy:
                            guidance_action = guidance_exp_strategy.return_action()
                            if guidance_exp_strategy.mistake:
                                supervisor_mistakes += 1

                            is_correction_performed = False

                            if not guidance_action == action:
                                is_correction_performed = True
                                corrections += 1

                            # Learning From Guidance - Shared Control Approach
                            action = guidance_action

                            result, next_state = self.get_next_state(state, states, normed_states, action,
                                                                     previous_result)
                            next_state_index = states.index(tuple(next_state))

                            reward = result if result > 0.0 else -1.0
                            engagement = self.get_engagement(normed_states[next_state_index], result)

                            if update_mode == 1:
                                reward += beta1 * engagement
                            elif update_mode == 2:
                                reward = beta2 * engagement

                            next_action = 0

                        Q[state_index][:], error = learning.update(state_index, action, next_state_index, next_action,
                                                                    reward, Q[state_index][:], Q[next_state_index][:], done)

                    ERROR.append(error)

                    state = next_state
                    previous_result = result

                episode += 1
                R.append(r)
                MS.append(max_score)
                CORRECTIONS.append(corrections)
                SUPERVISOR_MISTAKES.append(supervisor_mistakes)
                V.append(max(Q[start_state_index][:]))
                ER.append(np.asarray(ERROR).mean())
                S.append(score)
                ENG.append(np.asarray(EE).mean())

                logger.log_value(str(max(Q[start_state_index][:])) + '\n')
                logger.log_return(str(r) + '\n')
                logger.log_score(str(score) + '\n')
                logger.log_max_score(str(max_score) + '\n')
                logger.log_engagement(str(np.asarray(EE).mean()) + '\n')
                logger.log_error(str(np.asarray(ERROR).mean()) + '\n')
                if guidance_policy:
                    logger.log_corrections(str(corrections) + '\n')
                    logger.log_supervisor_mistakes(str(supervisor_mistakes) + '\n')

            logger.stop()

            self.save_plot(data=R, epochs=epochs, run=run, dir_name=name, data_name='return')
            self.save_plot(data=ENG, epochs=epochs, run=run, dir_name=name, data_name='engagement')
            self.save_plot(data=V, epochs=epochs, run=run, dir_name=name, data_name='mean_v(s)')
            self.save_plot(data=S, epochs=epochs, run=run, dir_name=name, data_name='score')
            self.save_plot(data=ER, epochs=epochs, run=run, dir_name=name, data_name='error')
            self.save_plot(data=zip(MS, S), epochs=epochs, run=run, dir_name=name, data_name='succes_ratio')

            if guidance_policy:
                self.save_plot(data=CORRECTIONS, epochs=epochs, run=run, dir_name=name, data_name='corrections')
                self.save_plot(data=SUPERVISOR_MISTAKES, epochs=epochs, run=run, dir_name=name,
                               data_name='supervisor_mistakes')

            self.save_qtable(Q, name, run)
            self.save_policy(states, exp_strategy, Q, name, run)
