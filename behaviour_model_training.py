#!/usr/bin/python

'''
    Copyright 2022 by Michał Stolarz <michal.stolarz@h-brs.de>

    This file is part of migrave_personalised_behaviour_model.
    It is used for running the complete pipeline for the training the behaviour model.

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

import sys
from behaviour_model.BehaviourModel import BehaviourModel
from behaviour_model_params import get_params

params = get_params()
user_id = params[2]
engagement_model_path = f"user_model/output/model/user{user_id}_engagement.json"
performance_model_path = f"user_model/output/model/user{user_id}_performance.json"

behaviour_model = BehaviourModel(performance_model_path, engagement_model_path, params)
behaviour_model.train()
