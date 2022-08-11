#!/usr/bin/python

import tensorflow as tf
import numpy as np

import itertools
from copy import deepcopy
from sklearn import metrics


def get_moments(x: np.array,
                axes: list = [0]):
    """
    Get mean and variance from the normalization
    :param x: Data to normalize
    :param axes: Axe to calculate mean and variance
    """
    mean, variance = tf.nn.moments(x, axes=axes)
    return mean, variance


def normalize_with_moments(x: np.array,
                           axes: list = [0],
                           epsilon: float = 1e-8,
                           mean: np.array = None,
                           variance: np.array = None):
    """
    Z normalization
    :param x: Data to normalize
    :param axes: Axe to calculate mean and variance
    :param epsilon: epsilon so that variance is never zero
    :param mean: Mean values (if they are known)
    :param variance: Variance valus (if they are known)
    :return: Normalized data
    """
    if mean is None or variance is None:
        mean, variance = get_moments(x, axes=axes)
    x_normed = (x - mean) / tf.sqrt(variance + epsilon)  # epsilon to avoid dividing by zero
    return x_normed

def grid_search(estimator, param_grid, x, y):
    """
    Grid search for hyperparameters search
    :param estimator: Estimator to tune
    :param param_grid: Grid with hyperparameters to try out
    :param x: Input training data
    :param y: Reference values for training the model
    """

    model={}
    keys = param_grid.keys()
    combs = [param_grid[key] for key in param_grid]
    params_set = list(itertools.product(*combs))
    old_error = 100.0

    for params in params_set:
        params_dict = {key: value for key, value in zip(keys, params)}
        estimator.set_params(**params_dict)
        estimator.fit(x, y)
        error = metrics.mean_squared_error(y, estimator.predict(x), squared=False)

        if old_error > error:
            estimator_cpy = deepcopy(estimator)
            model = {'model': estimator_cpy,
                     'score': error,
                     'params': params_dict}
            old_error = error
    return model
