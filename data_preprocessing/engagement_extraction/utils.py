from typing import Dict, Tuple
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

MIGRAVE_VISUAL_FEATURES = ['of_AU01_c', 'of_AU02_c', 'of_AU04_c', 'of_AU05_c',
                           'of_AU06_c', 'of_AU07_c', 'of_AU09_c', 'of_AU10_c', 'of_AU12_c',
                           'of_AU14_c', 'of_AU15_c', 'of_AU17_c', 'of_AU20_c', 'of_AU23_c',
                           'of_AU25_c', 'of_AU26_c', 'of_AU28_c', 'of_AU45_c', 'of_gaze_0_x',
                           'of_gaze_0_y', 'of_gaze_0_z', 'of_gaze_1_x', 'of_gaze_1_y',
                           'of_gaze_1_z', 'of_gaze_angle_x', 'of_gaze_angle_y', 'of_pose_Tx',
                           'of_pose_Ty', 'of_pose_Tz', 'of_pose_Rx', 'of_pose_Ry', 'of_pose_Rz']
NON_FEATURES_COLS = ["participant", "session_num", "timestamp", "engagement"]

# Cols to fill with max or min
NAN_MAX_COLS = ['of_gaze_0_x',
                'of_gaze_0_y',
                'of_gaze_0_z',
                'of_gaze_1_x',
                'of_gaze_1_y',
                'of_gaze_1_z',
                'of_gaze_angle_x',
                'of_gaze_angle_y',
                'of_gaze_distance',
                'of_gaze_distance_x',
                'of_gaze_distance_y',
                'of_pose_Rxv',
                'of_pose_Ry',
                'of_pose_Rz',
                'of_pose_Tx',
                'of_pose_Ty',
                'of_pose_Tz',
                'of_pose_distance']

# Some codes are based on
# https://github.com/interaction-lab/exp_engagement/tree/master/Models
def standardize_data(data: pd.core.frame.DataFrame,
                     mean: Dict[str, float] = None,
                     std: Dict[str, float] = None) -> Tuple[pd.core.frame.DataFrame,
                                                            Dict[str, float],
                                                            Dict[str, float]]:
    """Normalises each column with respect to the mean and standard deviation,
    and fills NaN values with the maximum column value. If mean and std are None,
    calculates the column means and standard deviations from the data; otherwise,
    uses the provided values for normalisation.
    Returns:
    * the normalised data
    * a dictionary of column names and mean values (the same as 'mean' if 'mean' is given)
    * a dictionary of column names and standard deviations (the same as 'std' if 'std' is given)
    Keyword arguments:
    @param data: pd.core.frame.DataFrame -- data to be normalised
    @param mean: Dict[str, float] -- dictionary of column names and column means
                                     (default None, in which case the means are
                                      calculated from the data)
    @param std: Dict[str, float] -- dictionary of column names and column standard deviations
                                    (default None, in which case the standard deviations are
                                     calculated from the data)
    """
    data_mean = {}
    data_std = {}
    data_copy = data.copy()
    for c in data.columns:
        col_mean = mean[c]
        col_std = std[c]

        data_mean[c] = col_mean
        data_std[c] = col_std

        if abs(col_std) < 1e-10:
            data_copy[c] = data_copy[c] - col_mean
        else:
            data_copy[c] = (data_copy[c] - col_mean) / col_std

        # fill nan with min if column not in NAN_MAX_COLS, otherwise fill with max
        if c not in NAN_MAX_COLS:
            min_val = np.nanmin(data_copy[c])
            data_copy[c] = data_copy[c].fillna(min_val)
        else:
            max_val = np.nanmax(data_copy[c])
            data_copy[c] = data_copy[c].fillna(max_val)

    return data_copy, data_mean, data_std

