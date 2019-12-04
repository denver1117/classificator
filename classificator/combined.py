"""
Combination estimator to load and use individual estimators as
inputs into a final tree based estimator
"""

import numpy as np
import random
import boto3
import pickle
import os

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

__all__ = ["CombinedEstimator"]

tmpdir = "/tmp"

class CombinedEstimator(TransformerMixin, BaseEstimator):
    """ 
    Estimator used to combine multiple model predictions 
    into a new feature space for prediction, and trains
    a random forest for final prediction
    """

    def __init__(self, model_locs=[], feature_indices=[],
            n_estimators=5, min_samples_split=2,
            max_features=1.0):
        """ Set inputs as attributes """

        self.model_locs = model_locs
        self.feature_indices = feature_indices
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.max_features = max_features

    def _load_model(self, loc):
        """ 
        Loads model either from local or S3 locations. Will return 
        intput if already a model object
        """

        # If not a string, return input
        if not (isinstance(loc, str) or isinstance(loc, unicode)):
            return loc

        # If location is in S3, copy to local, then unpickle 
        to_delete = False
        if "s3" in loc:
            tmp_loc = "{0}/tmp_file_{1}.obj".format(tmpdir, random.randint(1,1000))
            s3 = boto3.client('s3')
            bucket = loc.split("/")[2]
            key = "/".join(loc.split("/")[3:])
            with open(tmp_loc, "wb") as data:
                s3.download_fileobj(bucket, key, data)
            loc = tmp_loc
            to_delete = True
        with open(loc, "rb") as f:
            model = pickle.load(f)
        if to_delete:
            os.remove(tmp_loc)
        return model

    def _transform_models(self, X):
        """ 
        Runs predictions of input models on input data X and 
        stacks them into a single feature space
        """

        out = []
        for index in range(len(self.models)):
            arr = np.array([list(np.array(x)[self.feature_indices[index]]) for x in X])
            out.append(self.models[index].predict_proba(arr))
        return np.hstack(out)

    def fit(self, X, y=None):
        """ Fits final random forest on input data """
        
        self.models = [
            self._load_model(loc)
            for loc in self.model_locs]
        self.estimator = Pipeline(steps=[
            ("imputer", SimpleImputer()),
            ("clf", RandomForestClassifier(
                n_estimators=self.n_estimators, 
                min_samples_split=self.min_samples_split,
                max_features=self.max_features))
            ])
        
        probs = self._transform_models(X)
        self.estimator.fit(probs, y)
        self.classes_ = self.estimator.classes_
        return self

    def predict(self, X):
        """ Predict class """

        probs = self._transform_models(X)
        return self.estimator.predict(probs)

    def predict_proba(self, X):
        """ Predict probabilities """

        probs = self._transform_models(X)
        return self.estimator.predict_proba(probs)

    def score(self, X, y):
        """ Scores estimator """

        probs = self._transform_models(X)
        return self.estimator.score(probs, y)


