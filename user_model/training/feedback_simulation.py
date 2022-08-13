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

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np

from simulation.user_model.simulation_utils import normalize_with_moments, get_moments, grid_search
from simulation.user_model.simulation import Simulation
from simulation.user_model.models import FeedbackNN

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RationalQuadratic, RBF, WhiteKernel, Matern
from sklearn.svm import SVR
from sklearn import metrics
import json


class FeedbackSimulation(Simulation):
    def __init__(self, data):
        super().__init__(data)
        self.a0 = self.P0.groupby(['length', 'robot_feedback', 'previous_score', 'current_result'])
        self.a1 = self.P1.groupby(['length', 'robot_feedback', 'previous_score', 'current_result'])

    def train(self, model_name):
        """
        Train models on all clusters
        :param model_name: model to be the user performance simulation [rf, ab, svm, gp, nn], that is
        rf - random forest,
        ab - adaboost,
        svm - support vector machine,
        gp - gaussian process,
        nn - neural network
        :return: None
        """
        for ii, cluster in enumerate([self.a0, self.a1]):
            model = self.__train_cluster(cluster, model_name)
            self.models.append({"id": ii,
                                "model_name": model_name,
                                "model": model,
                                "cluster": cluster})
            if model_name == 'nn':
                model.save('simulation/output/model/user' + str(ii) + '_feedback.h5', 'wb')
            print(f"Trained cluster no {ii}")

    def eval(self, if_save=True):
        """
        Evaluate all trained models on the set of all possible states.
        :param if_save: flag if the evaluation results should be saved in the form of json
        :return: None
        """
        if not self.models:
            raise RuntimeError("The models have to be trained before ealuation")

        for ii, model in enumerate(self.models):
            self.__eval_cluster(**model, if_save=True)
            print(f"Evaluated cluster no {ii}")

    def __extract_data_from_cluster(self, cluster):
        """
        Extracting X and y data from the cluster
        :param cluster: Cluster of data
        :return: None
        """
        train_X = []
        train_Y = []
        for key, item in cluster:
            A = cluster.get_group(key)
            if A['engagement'].mean() < 1.0:
                training = [self.L[self.D.index(key[0])],
                            self.RF[key[1]][0],
                            self.RF[key[1]][1],
                            self.RF[key[1]][2],
                            self.PS[self.S.index(key[2])],
                            key[3]]
                target = A['engagement'].mean()
                train_X.append(training)
                train_Y.append(target)
        return train_X, train_Y

    def __train_cluster(self, cluster, model_name):
        """
        Train a chosen model [rf, ab, svm, gp, nn] on the data cluster
        :param cluster: cluster of the data
        :param model: model to be the user performance simulation [rf, ab, svm, gp, nn], that is
        rf - random forest,
        ab - adaboost,
        svm - support vector machine,
        gp - gaussian process,
        nn - neural network
        :return: None
        """
        train_X, train_Y = self.__extract_data_from_cluster(cluster)
        x = np.asarray(train_X)
        y = np.asarray(train_Y)
        x_train = normalize_with_moments(x)

        if model_name == "nn":
            model = FeedbackNN()
            history = model.fit(x_train, y, epochs=10000, batch_size=8, verbose=0)
            plt.plot(history.history['loss'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()
        elif model_name == "gp":
            kernel = RationalQuadratic(length_scale=1.0, alpha=1.5, length_scale_bounds=(1e-5, 1e5), alpha_bounds=(1e-5, 1e5))
            # kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale = .1, length_scale_bounds=[1e-2, 1e2]) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-4))
            model = GaussianProcessRegressor(kernel=kernel)
            model.fit(np.array(x_train), np.array(y))
            print("GP kernel params: ", model.kernel_.get_params())

        else:
            if model_name == "rf":
                random_grid = {'n_estimators': [int(x) for x in np.arange(start=50, stop=1000, step=50)]}
                model = RandomForestRegressor()
            elif model_name == "ab":
                random_grid = {'n_estimators': [int(x) for x in np.arange(start=50, stop=1000, step=50)]}
                model = AdaBoostRegressor()
            elif model_name == "svm":
                random_grid = {'kernel': ['poly', 'linear', 'rbf', 'sigmoid'],
                               'C': [0.1, 1, 10, 100, 1000, 10000],
                               'epsilon': [0.0001, 0.001, 0.01, 0.1]}
                model = SVR()

            clf = grid_search(estimator=model, param_grid=random_grid, x=np.array(x_train), y=np.array(y))
            model = clf['model']
            print(clf['params'])

        y_pred_test = model.predict(x_train)
        print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y, y_pred_test))
        print('Root Mean Squared Error (RMSE):', metrics.mean_squared_error(y, y_pred_test, squared=False))
        return model

    def __eval_cluster(self, id, cluster, model, model_name, if_save=True):
        """
        Evaluate single model (trained on the single cluster)
        on the set of all possible states
        :param id: id of the model
        :param cluster: cluster of the data on which the model was trained
        :param model: model trained on the specified cluster of data
        :param model_name: name of the trained model can be [rf, ab, svm, gp, nn]
        :param if_save: flag if the evaluation results should be saved in the form of json
        :return: None
        """
        matplotlib.rc('font', **{'size': 20})
        matplotlib.use("pgf")
        matplotlib.rcParams.update({
            "pgf.texsystem": "pdflatex",
            'font.family': 'serif',
            'text.usetex': True,
            'pgf.rcfonts': False,
        })

        train_X, train_Y = self.__extract_data_from_cluster(cluster)
        x = np.asarray(train_X)
        mean, variance = get_moments(x)

        preds_succ = []
        preds_failure = []
        args_failure = []
        args_succ = []
        stds_succ = []
        stds_failure = []

        fig_width = 10
        fig_height = 6
        figure(figsize=(fig_width, fig_height), dpi=400)

        print("Length of states: ", len(self.states))
        model_dict = {}
        model_dict_std = {}
        training_dict = {}

        for result in [-1, 1]:
            res = normalize_with_moments(result, mean=mean[-1], variance=variance[-1])
            for i, s in enumerate(normalize_with_moments(np.asarray(self.states),
                                                         mean=mean[:-1],
                                                         variance=variance[:-1])):
                if model_name == "nn":
                    pred = model.predict(np.asarray([s[0], s[1], s[2], s[3], s[4], res]).reshape(1, 6))[0][0]
                elif model_name == "gp":
                    pred, std = model.predict(np.asarray([s[0], s[1], s[2], s[3], s[4], res]).reshape(1, 6),
                                              return_std=True)
                    if result == 1:
                        stds_succ.append(float(std))
                    elif result == -1:
                        stds_failure.append(float(std))

                    model_dict_std[str((self.states[i][0], self.states[i][1], self.states[i][2],
                                        self.states[i][3], self.states[i][4], float(result)))] = float(std)
                else:
                    pred = model.predict(np.asarray([s[0], s[1], s[2], s[3], s[4], res]).reshape(1, 6))

                if pred > 1:
                    pred = 1
                elif pred < -1:
                    pred = -1

                model_dict[str((self.states[i][0], self.states[i][1], self.states[i][2],
                                self.states[i][3], self.states[i][4], float(result)))] = float(pred)

                if result == 1:
                    preds_succ.append(float(pred))
                    args_succ.append(i)
                elif result == -1:
                    preds_failure.append(float(pred))
                    args_failure.append(i)

        plotting_features = [("engagement (failure)", 'red', args_failure, preds_failure, stds_failure, -1),
                             ("engagement (success)", 'blue', args_succ, preds_succ, stds_succ, 1)]
        vmax = []

        for label, color, args, preds, stds, result in plotting_features:
            plt.plot(args, preds, label=f"predicted {label}", color=color)
            # vmax.append(max(preds))

            if model_name == "gp":
                preds_ = np.array(preds)
                stds_ = np.array(stds)

                pos_shift_preds = np.array([value if value <= 1 else 1 for value in list(preds_ + stds_)])
                neg_shift_preds = np.array([value if value >= -1 else -1 for value in list(preds_ - stds_)])

                plt.fill_between(np.array(args),
                                 y1=pos_shift_preds,
                                 y2=neg_shift_preds,
                                 color=f"tab:{color}",
                                 alpha=0.2)
            state_x = []
            state_y = []
            for a, b in zip(train_X, train_Y):
                if a[5] == result:
                    state_x.append(self.states.index(tuple([a[0], a[1], a[2], a[3], a[4]])))
                    state_y.append(b)
                    training_dict[str((a[0], a[1], a[2], a[3], a[4], a[5]))] = float(b)

            plt.plot(state_x, state_y, f"o{color[0]}", label=f"estimated {label}")

        for bound in [-1, 1]:
            plt.bar(self.state_level,
                    height=[bound*1.2] * len(self.state_level),
                    width=self.state_level_nums,
                    color=['w', 'k'] * int(len(self.state_level) / 2),
                    align="edge",
                    alpha=0.1)

        for i in range(len(self.state_level)):
            plt.text((self.state_level[i]+self.state_level_nums[i]/2 - 3), -0.95, f"Level {i+1}", bbox={'facecolor':'white','alpha':1,'edgecolor':'black','pad':3})

        plt.legend(borderaxespad=0)
        plt.grid()
        plt.xlim(0, len(self.states))
        plt.ylim(-1, 1)
        plt.xlabel("State id")
        plt.ylabel("Engagement level")
        plt.title(f"Engagement plot for user cluster {id}")
        plt.savefig(f"simulation/output/graphics/feedback_c{id}.pdf")
        plt.close()

        if if_save:
            with open(f"simulation/output/model/user{id}_feedback.json", "w") as model_json:
                json.dump(model_dict, model_json)
            with open(f"simulation/output/model/user{id}_feedback_training.json", "w") as model_json:
                json.dump(training_dict, model_json)
            if model_name == "gp":
                with open(f"simulation/output/model/user{id}_feedback_std.json", "w") as model_json:
                    json.dump(model_dict_std, model_json)