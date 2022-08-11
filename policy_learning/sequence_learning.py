#!/usr/bin/python
import numpy as np
from policy_learning.RL import MDP, Policy, Learning, Representation
import random
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import sys, os
import csv
import itertools
import random
from datetime import datetime
from options import GetOptions
import json


def load_model(path):
    with open(path) as model_json:
        model_dict = json.load(model_json)

    new_model_dict = {}
    for key in model_dict:
        new_key = key.replace("(", "")
        new_key = new_key.replace(")", "")
        tuple_key = tuple([float(item) for item in new_key.split(',')])
        new_model_dict[tuple_key] = model_dict[key]

    return new_model_dict


# define state-action space
def state_action_space():
    length = [3, 5, 7]
    feedback = [0, 1, 2]
    previous = [-3, -2, -1, 0, 1, 2, 3]

    combs = (length, feedback, previous)
    states = list(itertools.product(*combs))
    states.append((0, 0, 0))

    l = [1, 2, 3]
    f = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    combs = (l, f, previous)
    normalized_states = list(itertools.product(*combs))
    normalized_states.append((0, [0, 0, 0], 0))

    actions = [0, 1, 2, 3, 4]
    actions_oh = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]
    return states, normalized_states, actions, actions_oh


def scale(value, limits=(-1, 1), desired_limits=(0, 1)):
    if not isinstance(value, float):
        raise ValueError("The given value should be of type float")

    if limits[0] >= limits[1] or desired_limits[0] >= desired_limits[1]:
        raise ValueError("Bottom limit can not be greater or equal to the upper limit")
    return np.interp(value, limits, desired_limits)


def get_engagement(state, result, model):
    outcome = 1 if result > 0 else -1
    st = tuple([state[0] / 3.0, state[1][0], state[1][1], state[1][2], state[2] / 3.0, outcome])
    engagement = model[st]
    return engagement


def get_diff(state, result, action, model):
    outcome = 1 if result > 0 else -1

    st = state[0], state[1][0], state[1][1], state[1][2], state[2], outcome, action[0], action[1], action[2], action[3], \
         action[4], action[5]

    engagement = model.predict(np.asarray(st).reshape(1, 6))[0]
    return engagement


def get_next_state(state, states, normalized_states, action, previous, model):
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
    prob = model[st]

    # print st, prob

    if random.random() <= prob:
        success = 1
    else:
        success = -1

    score = success * levels[length]
    return score, [length, feedback, previous]


def moving_average(a, n=200):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


episodes, epochs, user, q_table, name, learn, interactive_type, To, alpha, gamma, \
update_mode, beta1, beta2, exploration_policy, guidance_policy, runs_num, \
p_guidance_mistakes, reward_function = GetOptions(sys.argv[1::])

if not os.path.exists(f"results/{name}"):
    os.makedirs(f"results/{name}")
    os.makedirs(f"results/{name}/runs")

for run in range(runs_num):
    if not os.path.exists(f"results/{name}/runs/{run}"):
        os.makedirs(f"results/{name}/runs/{run}")

    g = open(f"results/{name}/runs/{run}/episodes", 'w')
    rr = open(f"results/{name}/runs/{run}/return", 'w')
    ss = open(f"results/{name}/runs/{run}/score", 'w')
    ms = open(f"results/{name}/runs/{run}/max_score", 'w')
    vv = open(f"results/{name}/runs/{run}/v_start", 'w')
    ee = open(f"results/{name}/runs/{run}/engagement", 'w')
    er = open(f"results/{name}/runs/{run}/error", 'w')
    corr = open(f"results/{name}/runs/{run}/corrections", 'w')
    mist = open(f"results/{name}/runs/{run}/supervisor_mistakes", 'w')

    logfile = open(f"results/{name}/runs/{run}/logfile", 'w')
    logfile.write(f"logfile for: {name} - {datetime.now()} \n\n")
    logfile.write(f"Episodes: {episodes} \n")
    logfile.write(f"Epochs: {epochs} \n")
    logfile.write(f"User: {user} \n")
    logfile.write(f"Qtable: {q_table} \n")
    logfile.write(f"Learning: {learn} \n")
    logfile.write(f"Learning Rate: {alpha} \n")
    logfile.write(f"Discount factor: {gamma} \n")
    logfile.write(f"Interactive {interactive_type} \n\n")
    logfile.write(f"Exploration: {To} \n")
    logfile.write(f"Update mode: {update_mode} \n")
    logfile.write(f"Exploration policy: {exploration_policy} \n")
    logfile.write(f"Guidance policy: {guidance_policy} \n")
    logfile.write(f"Probability of guidance mistakes: {p_guidance_mistakes} \n")
    logfile.write(f"Reward function: {reward_function} \n")
    logfile.write(f"Numbers of runs: {runs_num} \n")
    logfile.write(f"beta1: {beta1} \n")
    logfile.write(f"beta2: {beta2} \n")
    logfile.close()

    cmodel = load_model(f"../simulation/output/model/user{user}_feedback.json")
    model = load_model(f"../simulation/output/model/user{user}_performance.json")

    # start and terminal states and indices
    states, normed_states, actions, actions_oh = state_action_space()
    A = ['L = 3', 'L = 5', 'L = 7', 'PF', 'NF']

    first_length = random.choice([3, 5, 7])
    start_state = (0, 0, 0)
    start_state_index = states.index(tuple(start_state))

    # define MDP and policy
    m = MDP(start_state, actions)
    m.states = states

    table = Representation('qtable', [m.actlist, m.states])
    pretrained = False

    Q_guidance = np.asarray(table.Q)
    if guidance_policy:
        print('Loading Q-table guidance policy: ' + str(guidance_policy))
        ins = open(guidance_policy, 'r')
        Q_guidance = np.array([[float(n) for n in line.split()] for line in ins])
        ins.close()

    Q = np.asarray(table.Q)
    if q_table:
        print('Loading Q-table: ' + str(q_table))
        pretrained = True
        ins = open(q_table, 'r')
        Q = np.array([[float(n) for n in line.split()] for line in ins])
        ins.close()
    table.Q = Q

    if guidance_policy:
        egreedy = Policy(name="exploitation", param=To)
        guidance_egreedy = Policy(name="guidance", param=To, p_guidance_mistakes=p_guidance_mistakes)
    else:
        egreedy = Policy(name=exploration_policy, param=To)

    # alpha = float(0.01)
    #gamma = float(0.95)
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
            g.write('Episode No.' + str(episode) + '\n')
            print('Episode No.' + str(episode) + '\n')

        while (not done):
            state_index = states.index(tuple(state))
            egreedy.Q_state = Q[state_index][:]

            if guidance_policy:
                guidance_egreedy.Q_state = Q_guidance[state_index][:]

            # adaptive exploration per state visit
            egreedy.param = To - 5 * float(visits[state_index])

            if egreedy.param < 0.5:
                egreedy.param = 0.5

            if episode % epochs == 0:
                visits[state_index] += 1

            # robot feedback (actions 3,4) is not available on first state
            if state_index == start_state_index:
                egreedy.Q_state = Q[state_index][:3]
                if guidance_policy:
                    guidance_egreedy.Q_state = Q_guidance[state_index][:3]

            action = egreedy.return_action()

            result, next_state = get_next_state(state, states, normed_states, action, previous_result, model)
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
            engagement = get_engagement(normed_states[next_state_index], result, cmodel)
            EE.append(engagement)

            if update_mode == 1:
                reward += beta1 * engagement
            elif update_mode == 2:
                reward = beta2 * engagement

            r += (learning.gamma ** (iteration - 1)) * reward

            # sarsa
            #egreedy.Q_next_state = Q[next_state_index][:]
            #next_action = egreedy.return_action()
            next_action = 0 #Because we Q-learning is used

            if episode % epochs == 0 or episode == 1:
                g.write(str(iteration) + '... ' + str(state) + ' ' + str(A[action]) +
                        ' ' + str(next_state) + ' ' + str(reward) + ' ' + str(score) +
                        ' ' + str(engagement) + '\n')

            if iteration == N:
                done = 1

            iteration += 1

            error = 0
            if learn:
                if guidance_policy:
                    guidance_action = guidance_egreedy.return_action()
                    if guidance_egreedy.mistake:
                        supervisor_mistakes += 1

                    is_correction_performed = False

                    if not guidance_action == action:
                        is_correction_performed = True
                        corrections += 1

                    # Learning From Guidance - Shared Control Approach
                    action = guidance_action

                    result, next_state = get_next_state(state, states, normed_states, action, previous_result, model)
                    next_state_index = states.index(tuple(next_state))

                    reward = result if result > 0.0 else -1.0
                    engagement = get_engagement(normed_states[next_state_index], result, cmodel)

                    if update_mode == 1:
                        reward += beta1 * engagement
                    elif update_mode == 2:
                        reward = beta2 * engagement

                    # sarsa
                    #egreedy.Q_next_state = Q[next_state_index][:]
                    #next_action = egreedy.return_action()
                    #guidance_egreedy.Q_next_state = Q_guidance[next_state_index][:]
                    #next_action = guidance_egreedy.return_action()
                    next_action = 0 #Because Q-learning is used

                    #if is_correction_performed and pretrained:
                        # Q-augmentation for feedback -- after update - DOES NOT WROK AS EXPECTED
                    #    reward = 1./alpha

                #else:
                Q[state_index][:], error = learning.update(state_index, action, next_state_index, next_action,
                                                            reward, Q[state_index][:], Q[next_state_index][:], done)

            ERROR.append(error)

            # Q-augmentation for engagement -- after update
            # if interactive_type:
            #     Q[state_index][action] += beta2 * engagement

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

        vv.write(str(max(Q[start_state_index][:])) + '\n')
        rr.write(str(r) + '\n')
        ss.write(str(score) + '\n')
        ms.write(str(max_score) + '\n')
        ee.write(str(np.asarray(EE).mean()) + '\n')
        er.write(str(np.asarray(ERROR).mean()) + '\n')
        if guidance_policy:
            corr.write(str(corrections) + '\n')
            mist.write(str(supervisor_mistakes) + '\n')

    print(visits)

    with open(f"results/{name}/runs/{run}/q_table", 'w') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerows(Q)

    figure(figsize=(10, 6), dpi=400)

    tmp = []
    return_epoch = []
    for i, t in enumerate(R):
        tmp.append(t)
        if i % epochs == 0:
            a = np.asarray(tmp)
            return_epoch.append(a.mean())
            tmp = []

    plt.plot(return_epoch)
    plt.grid()
    plt.xlabel("Epochs")
    plt.ylabel("Return")
    plt.savefig(f"results/{name}/runs/{run}/return.png")
    plt.close()

    figure(figsize=(10, 6), dpi=400)

    tmp = []
    eng_epoch = []
    for i, t in enumerate(ENG):
        tmp.append(t)
        if i % epochs == 0:
            a = np.asarray(tmp)
            eng_epoch.append(a.mean())
            tmp = []

    plt.plot(eng_epoch)
    plt.grid()
    plt.xlabel("Epochs")
    plt.ylabel("Engagement")
    plt.savefig(f"results/{name}/runs/{run}/engagement.png")
    plt.close()

    figure(figsize=(10, 6), dpi=400)

    tmp = []
    v_epoch = []
    for i, t in enumerate(V):
        tmp.append(t)
        if i % epochs == 0:
            a = np.asarray(tmp)
            v_epoch.append(a.mean())
            tmp = []

    plt.plot(v_epoch)
    plt.grid()
    plt.xlabel("Epochs")
    plt.ylabel("Mean V(s)")
    plt.savefig(f"results/{name}/runs/{run}/mean_v(s).png")
    plt.close()

    figure(figsize=(10, 6), dpi=400)

    tmp = []
    score_epoch = []
    for i, t in enumerate(S):
        tmp.append(t)
        if i % epochs == 0:
            a = np.asarray(tmp)
            score_epoch.append(a.mean())
            tmp = []

    plt.plot(score_epoch)
    plt.grid()
    plt.xlabel("Epochs")
    plt.ylabel("User acc points")
    plt.savefig(f"results/{name}/runs/{run}/score.png")
    plt.close()

    figure(figsize=(10, 6), dpi=400)

    tmp = []
    error_epoch = []
    for i, t in enumerate(ER):
        tmp.append(t)
        if i % epochs == 0:
            a = np.asarray(tmp)
            error_epoch.append(a.mean())
            tmp = []

    plt.plot(error_epoch)
    plt.grid()
    plt.xlabel("Epochs")
    plt.ylabel("Error update")
    plt.savefig(f"results/{name}/runs/{run}/error.png")
    plt.close()

    figure(figsize=(10, 6), dpi=400)

    tmp = []
    score_epoch = []
    for i, (maxs, s) in enumerate(zip(MS, S)):
        tmp.append(s / maxs)
        if i % epochs == 0:
            a = np.asarray(tmp)
            score_epoch.append(a.mean())
            tmp = []

    plt.plot(score_epoch)
    plt.grid()
    plt.xlabel("Epochs")
    plt.ylabel("Success ratio")
    plt.savefig(f"results/{name}/runs/{run}/succes_ratio.png")
    plt.close()

    if guidance_policy:
        figure(figsize=(10, 6), dpi=400)

        tmp = []
        corrections_epoch = []
        for i, correct in enumerate(CORRECTIONS):
            tmp.append(correct)
            if i % epochs == 0:
                a = np.asarray(tmp)
                corrections_epoch.append(a.mean())
                tmp = []

        plt.plot(corrections_epoch)
        plt.grid()
        plt.xlabel("Epochs")
        plt.ylabel("Corrections")
        plt.savefig(f"results/{name}/runs/{run}/corrections.png")
        plt.close()

        figure(figsize=(10, 6), dpi=400)

        tmp = []
        mistakes_epoch = []
        for i, mistake in enumerate(SUPERVISOR_MISTAKES):
            tmp.append(mistake)
            if i % epochs == 0:
                a = np.asarray(tmp)
                mistakes_epoch.append(a.mean())
                tmp = []

        plt.plot(mistakes_epoch)
        plt.grid()
        plt.xlabel("Epochs")
        plt.ylabel("Mistakes")
        plt.savefig(f"results/{name}/runs/{run}/supervisor_mistakes.png")
        plt.close()

    pf = open(f"results/{name}/runs/{run}/policy", 'w')
    for s, q in zip(states, Q):
        state_index = states.index(tuple(s))
        # argmaxQ = np.argmax(Q[state_index][:])
        # pf.write(str(state_index) + ' ' + str(s) + ' ' + str(argmaxQ) + '\n')
        pf.write(str(state_index) + ' ' + str(s))
        print(q)
        for i in q:
            softm = (np.exp(i / egreedy.param) / np.sum(np.exp(q / egreedy.param)))
            pf.write(' ' + str(softm))
        pf.write('\n')
    pf.close()

    vv.close()
    ss.close()
    ms.close()
    ee.close()
    er.close()
    g.close()
    rr.close()
    corr.close()
    mist.close()
