#!/usr/bin/python
import pandas as pd
from migrave_data.simulation.user_model.performance_simulation import PerformanceSimulation
from migrave_data.simulation.user_model.feedback_simulation import FeedbackSimulation

available_models = ["nn", "gp", "rf", "svm", "ab"]
performance_model_name = available_models[1]
feedback_model_name = available_models[1]

data = pd.read_csv('migrave_data/simulation/data/final_clustered.csv', delimiter=',')
C = data[['cluster', 'engagement', 'length', 'robot_feedback', 'previous_score', 'current_score', 'current_result',
          'action']]

performance_model = PerformanceSimulation(C)
performance_model.train(performance_model_name)
performance_model.eval(if_save=True)

feedback_model = FeedbackSimulation(C)
feedback_model.train(feedback_model_name)
feedback_model.eval(if_save=True)
