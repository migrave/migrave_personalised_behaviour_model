import os

import joblib
import numpy as np
from utils import standardize_data

class PersonStateEstimation(object):
    """
    Person state estimation. Engagement is estimated using xgboost
    """

    def __init__(self, config, pkg_dir):
        self._config = config
        self._engagement_cls = 0
        self._engagement_mea = 0
        self._engagement_std = 0
        #if "engagement" in self._config:
        #    cls_path = os.path.join(pkg_dir,
        #                            "models",
        #                            self._config["engagement"]["model_file"])

        #    self._engagement_cls, self._engagement_mean, self._engagement_std = self.load_classifier(cls_path)

    def load_classifier(self, cls_path):
        with open(cls_path, 'rb') as f:
            classifier, mean, std = joblib.load(f)

        self._engagement_cls = classifier
        self._engagement_mean = mean
        self._engagement_std = std

        return classifier, mean, std

    def estimate_engagement(self, features, normalize=True):
        """
        Estimate engagement given the current face features, audio signal,
        and game performance and return engagement scores

        :param features:       Face, audio, or game performance features
        :type name:            numpy.array

        :return:               Engagement
        """

        if normalize:
            features, mean, std = standardize_data(features, self._engagement_mean, self._engagement_std)
            #features = features.to_numpy()[0]
        
        probabilities = self._engagement_cls.predict_proba(features)
        max_index = np.argmax(probabilities, axis=1)
        prediction = self._engagement_cls.classes_[max_index]

        return prediction, probabilities[max_index]

