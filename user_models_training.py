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

import pandas as pd
from simulation.user_model.performance_simulation import PerformanceSimulation
from simulation.user_model.feedback_simulation import FeedbackSimulation

available_models = ["nn", "gp", "rf", "svm", "ab"]
performance_model_name = available_models[1]
feedback_model_name = available_models[1]

data = pd.read_csv('user_model/data/final_clustered.csv', delimiter=',')
C = data[['cluster', 'engagement', 'length', 'robot_feedback', 'previous_score', 'current_score', 'current_result',
          'action']]

performance_model = PerformanceSimulation(C)
performance_model.train(performance_model_name)
performance_model.eval(if_save=True)

feedback_model = FeedbackSimulation(C)
feedback_model.train(feedback_model_name)
feedback_model.eval(if_save=True)
