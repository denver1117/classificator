"""
Classificator class used to generate, train and validate a scikit-learn ML pipeline.
"""

import json
import uuid
import pandas as pd
import numpy as np
import boto3
import pickle
import datetime
import os
import shutil

from scipy.sparse import hstack, csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectFpr
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import KFold, GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, 
    roc_auc_score)
from functools import reduce

from .combined import CombinedEstimator
from .models import (
    scores, vectorizers, feature_selectors, 
    require_dense, pre_processors, 
    classifiers, datasets
    )

__all__ = ["Classificator"]

# directory names
logdir = "/tmp"
tmpdir = "/tmp"

class ClassificatorBase():
    """ Base class for Classificator """
    def __init__(self, config_loc="config.json", config=None):
        """ Accept configuration as input or load from text file """

        self.config_loc = config_loc
        if config:
            self.config = config
        else:
            self.config = self.load_config()
        if "run_name" not in self.config["data_specs"].keys():
            self.run_name = str(uuid.uuid1())
        else:
            self.run_name = self.config["data_specs"]["run_name"]
        self.log_name = "{0}/{1}.txt".format(logdir, self.run_name)
        if ("s3://" not in self.config["data_specs"]["out_loc"].lower()) and (not os.path.exists(self.config["data_specs"]["out_loc"])):
            raise IOError(
                "Output location: '{0}' not found".format(
                self.config["data_specs"]["out_loc"]))

    def load_config(self):
        """ Load configuration from text file """

        json_data=open(self.config_loc).read()
        return json.loads(json_data)

class Classificator(ClassificatorBase):
    """ 
    Generates multiple ML pipelines and chooses best model with
    grid search on each, using a common random state for standardizaiton
    of cross validation, and compares models using user defined scoring
    function. Saves best model to desired location along with
    validation statistics.
    """

    def choose_model(self):
        """ 
        Chooses the best model using multi-model grid search
        and cross validation.
        """

        # Initialize log file
        with open(self.log_name, "w"):
            pass
        os.system("sudo chmod 777 {0}".format(self.log_name))
 
        # Dump config and write to log
        with open("{0}/config.json".format(tmpdir), 'w') as f:
            json.dump(self.config, f)
        self._write("config.json")
        self.realtimelog("Starting") 

        # Deal with feature groups
        features = self.config["data_specs"]["feature_columns"]
        feature_methods = self.config["data_specs"]["feature_methods"]
        train_final = self.config["meta_specs"]["train_final"]
        if len(features) == 0:
            raise Exception("No feature columns supplied")
        elif isinstance(features[0], str) or isinstance(features[0], unicode):
            winner = self.get_best_model(stage="main")
        else:
            self.config["meta_specs"]["train_final"] = False
            model_list = []
            for index in range(len(features)):
                self.config["data_specs"]["feature_columns"] = features[index]
                self.config["data_specs"]["feature_methods"] = feature_methods[index]
                model_list.append(self.get_best_model(stage=["A", "B", "C"][index]))
            self.config["data_specs"]["feature_columns"] = list(np.sort(list(set([item for sublist in features for item in sublist]))))
            feature_indices = [[self.config["data_specs"]["feature_columns"].index(y) for y in x] for x in features]
            winner = CombinedEstimator(model_locs=model_list, feature_indices=feature_indices)
            (X, y, X_train, X_test, y_train, y_test,
                groups_train, groups_test) = self._prepare_input_data()
            winner.fit(X_train, y_train)
            self.get_stats(winner, X_test, y_test, stage="combined")

            # Retrain with full set if desired
            if train_final:
                winner.fit(X,y)

            # Add columns attribute for winner to maintain order of
            # feature labels for prediction API
            winner.features = self.config["data_specs"]["feature_columns"]

        # Pickle and save model
        if "model_name" in self.config["data_specs"].keys():
            pickle.dump(
                winner,
                open("{0}/{1}".format(tmpdir, self.config["data_specs"]["model_name"]), "wb"),
                pickle.HIGHEST_PROTOCOL)
            self._write(self.config["data_specs"]["model_name"])
        self.realtimelog("Pickled model object written to: {0}/{1}".format(
                self.config["data_specs"]["out_loc"],
                self.config["data_specs"]["model_name"]))
        self.realtimelog("CV stats, hold out stats, and model run configuration written to: {0}/".format(
                self.config["data_specs"]["out_loc"]))
        self.realtimelog("Finished")

    def get_best_model(self, stage="main"):
        # Train test split
        (X, y, X_train, X_test, y_train, y_test, 
            groups_train, groups_test) = self._prepare_input_data()

        # Generate pipelines and grids from config
        self._train_vectorizers(X_train)
        self._assemble_pipelines()
        self._assemble_grids()

        # Iterate through model choices, fitting a 
        # GridSearchCV model for each
        model_list = []
        scores_ = []
        for index in range(len(self.pipelines)):
            self.realtimelog(list(self.config["classifiers"].keys())[index], stage=stage)
            scoring = None
            if "score" in self.config["meta_specs"]:     
                scoring = scores[self.config["meta_specs"]["score"]]
            model = self._model_selection(
                X_train, y_train, self.pipelines[index], 
                self.grids[index], groups=groups_train, 
                k=self.config["meta_specs"]["k"],
                scoring=scoring)
            scores_.append(model.best_score_)
            model_list.append(model)
            self.realtimelog("CV score: {0}".format(model.best_score_), stage=stage)
            filename = "{0}_{1}.tsv".format(
                list(self.config["classifiers"].keys())[index].lower().replace(" ", "_"), 
                stage)
            pd.DataFrame(model.cv_results_).to_csv(
                "{0}/{1}".format(tmpdir, filename), 
                sep="\t", index=False)
            self._write(filename)

        # Choose winner
        winner_index = np.argmax(scores_)
        winner = model_list[winner_index].best_estimator_
        self.realtimelog("Best model type: {0}".format(
            list(self.config["classifiers"].keys())[winner_index]), stage=stage)

        # Get and write model validation statistics
        self.get_stats(winner, X_test, y_test, stage=stage)

        # Retrain with full set if desired
        if self.config["meta_specs"]["train_final"]:
            winner.fit(X,y)

        # Add columns attribute for winner to maintain order of
        # feature labels for prediction API
        winner.features = self.config["data_specs"]["feature_columns"]
        return winner

    def get_stats(self, winner, X_test, y_test, stage="main"):

        # Hold out validation with winner and metrics logging
        y_pred = winner.predict(X_test) 

        # Write hold out data
        hold_out_df = pd.DataFrame(
            np.hstack([X_test, [[x] for x in y_test], [[x] for x in y_pred]]), 
            columns=(
                self.config["data_specs"]["feature_columns"] + 
                ["label_true", "label_pred"]))
        hold_out_name = "hold_out_data.tsv"
        hold_out_df.to_csv("{0}/{1}".format(tmpdir, hold_out_name),
                sep="\t", index=False)
        self._write(hold_out_name)

        # Get accuracy score and classification report
        acc = accuracy_score(y_test, y_pred)
        clf_report = classification_report(y_test, y_pred, labels=np.unique(y_pred))
        clf_report_name = "clf_report_{0}.tsv".format(stage)
        classifaction_report_df(clf_report).to_csv(
                "{0}/{1}".format(tmpdir, clf_report_name),
                sep="\t", index=False)
        self._write(clf_report_name)
        self.realtimelog("Hold out classification report:", stage=stage)
        self.realtimelog(clf_report, stage=stage)
        self.realtimelog("Hold out ACC: {0}".format(acc), stage=stage)
        if not self.multiclass:
            try:
                y_prob = winner.predict_proba(X_test)[:,1]
                auc_roc = roc_auc_score(y_test, y_prob)
                self.realtimelog("Hold out ROC AUC: {0}".format(auc_roc), stage=stage)
            except:
                pass

        # Write metrics to file
        with open("{0}/hold_out_{1}.txt".format(tmpdir, stage), "w") as f:
            f.write("Hold out classification report:\n")
            f.write(clf_report)
            f.write("\n")
            f.write("Hold out ACC: {0}\n".format(acc))
            try:
                f.write("Hold out ROC AUC: {0}\n".format(auc_roc))
            except:
                pass
        self._write("hold_out_{0}.txt".format(stage))

    def realtimelog(self, log_str, stage="main"):
        """ Write string to log file """

        now = datetime.datetime.now()
        with open(self.log_name, "a") as f:
            f.write("{0} - {1} - {2}\n".format(now, stage, log_str))

    def _train_vectorizers(self, X):
        """ Train vectorizer models """

        self.vectorizers = []
        for col in range(len(self.config["data_specs"]["feature_methods"])):
            method = self.config["data_specs"]["feature_methods"][col]
            if method == "vectorize":
                tf = vectorizers[self.config["vectorizer"]["model"]](
                    **self.config["vectorizer"]["args"])
                tf.fit([force_str(x[col]) for x in X])
                vec = Vec_(tf)
            else:
                vec = None
            self.vectorizers.append(vec)

    def _write(self, filename):
        """ 
        Write temporary file to requested location either
        locally or in S3
        """

        loc = self.config["data_specs"]["out_loc"] 
        if "s3://" in loc.lower():
            s3 = boto3.resource('s3')
            splitted = loc.split("/")
            bucket = splitted[2]
            key = "/".join(splitted[3:])
            key_divider = "/" if splitted[-1] else ""
            destination = "{0}{1}{2}".format(key, key_divider, filename)
            if filename.split(".")[-1] in ["obj", "json"]:
                with open("{0}/{1}".format(tmpdir, filename), "rb") as data:
                    s3.meta.client.upload_fileobj(data, bucket, destination)
            else:
                s3.meta.client.upload_file("{0}/{1}".format(tmpdir, filename), bucket, destination)
        else:
            shutil.copyfileobj(
                open("{0}/{1}".format(tmpdir, filename), "rb"), 
                open("{0}/{1}".format(
                    loc[:-1] if loc[-1] == "/" else loc, 
                    filename), "wb")) 
        os.remove("{0}/{1}".format(tmpdir, filename))

    def _prepare_input_data(self):
        """ Loads and formats input data and performs train/test split """

        # Load from sklearn datasets or custom file source
        if self.config["data_specs"]["loc"].lower() in datasets.keys():
            df = load_sklearn_dataset(self.config["data_specs"]["loc"].lower())
        else:
            if ("sep" in self.config["data_specs"].keys()) and (self.config["data_specs"]["sep"] != ""):
                df = pd.read_csv(
                    self.config["data_specs"]["loc"], 
                    sep=self.config["data_specs"]["sep"])
            else:
                df = pd.read_csv(self.config["data_specs"]["loc"], sep=None)

        # Remove null labels
        df = df[~pd.isnull(df[self.config["data_specs"]["label_column"]])]

        # Set y for full set
        y = df[self.config["data_specs"]["label_column"]].values

        # Assign multiclass status
        if len(np.unique(y)) > 2:
            self.multiclass = True
            if self.config["meta_specs"]["score"] == "F1 (binary)":
                self.config["meta_specs"]["score"] = "F1 (macro)"
        else:
            self.multiclass = False

        # Handles grouping
        if len(self.config["data_specs"]["group_columns"]) == 1:
            groups = df[self.config["data_specs"]["group_columns"][0]].values
        else:
            groups = None

        # Set X for full set and split for train/test split
        X = df[self.config["data_specs"]["feature_columns"]].values
        if groups is not None:
            X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
                X, y, groups, test_size=(1-self.config["meta_specs"]["split_ratio"]), 
                random_state=self.config["meta_specs"]["random_seed"])
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=(1-self.config["meta_specs"]["split_ratio"]),
                random_state=self.config["meta_specs"]["random_seed"])
            groups_train = None
            groups_test = None
        return X, y, X_train, X_test, y_train, y_test, groups_train, groups_test

    def _assemble_pipelines(self):
        """ Iterate through pipeline generator for each estimator """

        self.pipelines = []
        for clf in self.config["classifiers"].keys():
            self.pipelines.append(self._assemble_pipeline(classifiers[clf](), clf))

    def _assemble_pipeline(self, clf, clf_name):
        """ 
        Assemble vectorization and ML pipeline for the given 
        estimator and configuration 
        """

        if self.config["meta_specs"]["bypass"] == 1:
            return clf

        if "selector" in self.config.keys():
            selector = feature_selectors[self.config["selector"]["model"]]
        else:    
            selector = None
        mtv = MultiTextVectorizer(
            self.config, 
            selector=selector)
        if clf_name in require_dense:
            clf_steps = [("densify", Densify()), ("clf", clf)]
        else:
            clf_steps = [("clf", clf)]
        steps = []
        steps.append(("vec", mtv))
        if "pre_processors" in self.config.keys():
            steps.append(("normify", Normify()))
        steps = steps + clf_steps
        if self.multiclass:
            steps = [("ovr", OneVsRestClassifier(Pipeline(steps=steps)))]

        # Sklearn pipeline instantiation
        out = Pipeline(steps=steps)
        return out

    def _assemble_grids(self):
        """ Iterate through grid generator for each estimator """

        self.grids = []
        for clf_name in self.config["classifiers"].keys():
            self.grids.append(self._assemble_grid(clf_name))
        
    def _assemble_grid(self, clf_name):
        """ Assemble nested grid for GridSearchCV parameter map """

        # Grid for estimator
        clf_grid = self.config["classifiers"][clf_name]
        if self.config["meta_specs"]["bypass"] == 1:
            return clf_grid
        new_clf_grid = {}
        for key in clf_grid.keys():
            if self.multiclass:
                new_name = "ovr__estimator__clf__{0}".format(key)
            else:
                new_name = "clf__{0}".format(key)
            new_clf_grid[new_name] = clf_grid[key]

        # Grid for vectorizers
        if self.multiclass:
            new_clf_grid["ovr__estimator__vec__vecs"] = [self.vectorizers]
        else:
            new_clf_grid["vec__vecs"] = [self.vectorizers]

        # Grid for pre-processeros
        if "pre_processors" in self.config.keys():
            proc_grid = self.config["pre_processors"]
            new_proc_grid = {}
            for key in proc_grid.keys():
                if self.multiclass:
                    new_name = "ovr__estimator__normify__{0}".format(key)
                else:
                    new_name = "normify__{0}".format(key)
                new_proc_grid[new_name] = proc_grid[key]
            new_clf_grid = reduce(lambda x,y: dict(x, **y), (new_clf_grid, new_proc_grid))

        # Grid for feature selection
        if "selector" in self.config.keys():
            selec_grid = self.config["selector"]["grid"]
            new_selec_grid = {}
            for key in selec_grid.keys():
                if self.multiclass:
                    new_name = "ovr__estimator__vec__{0}".format(key)
                else:
                    new_name = "vec__{0}".format(key) 
                new_selec_grid[new_name] = selec_grid[key]
            new_clf_grid = reduce(lambda x,y: dict(x, **y), (new_clf_grid, new_selec_grid)) 
        return new_clf_grid  

    def _model_selection(self, X_train, y_train, model, grid, groups=None, k=5, scoring=None):
        """ Run GridSearchCV fit method """
 
        # Use GroupKFold or KFold
        if groups is not None:
            kf = GroupKFold(n_splits=k)
            cv = kf.split(X_train, y_train, groups)
        else:
            kf = KFold(
                n_splits=k, shuffle=True, 
                random_state=self.config["meta_specs"]["random_seed"])
            cv = kf.split(X_train, y_train)
        clf = GridSearchCV(
            model, grid, cv=cv, 
            scoring=scoring, n_jobs=self.config["meta_specs"]["n_jobs"]) 
        clf.fit(X_train, y_train)
        return clf

#########################################
### Utility Classes for Classificator ###
#########################################        

class MultiTextVectorizer(TransformerMixin, BaseEstimator):
    """ Transform input data column according to method specified in config """

    def __init__(self, config, vecs=None, selector=None, alpha=0.05):
        """ Set input config and kwargs as attributes """

        self.config = config
        self.vecs = vecs
        self.selector = selector
        self.alpha = alpha
        self.num_methods = len(
            self.config["data_specs"]["feature_methods"])

    def fit(self, X, y=None):
        """ Fit vectorizers """

        self.vectorizers = []
        for col in range(self.num_methods):
            method = self.config["data_specs"]["feature_methods"][col]
            if method == "vectorize":
                tf = self.vecs[col]
                if self.selector:
                    selector = SelectFpr(self.selector, alpha=self.alpha)
                    selector.fit(tf.transform(self._get_col(X, col, method)), y=y)
                    vec = Pipeline(
                        steps=[("tf", tf), ("select", selector)])
                else:
                    vec = self.vecs[col]
            elif method == "encode":
                vec = Encoder()
                vec.fit(self._get_col(X, col, method))
            elif method == "numeric":
                vec = Sparsify(scale=True)
                vec.fit(self._get_col(X, col, method))
            else:
                raise ValueError("Unknown feature method: {0}".format(method)) 
            self.vectorizers.append(vec)
        return self

    def transform(self, X): 
        """ Transform input data with fitted vectorizers """

        transformed = []
        for col in range(self.num_methods):
            vec = self.vectorizers[col].transform(
                self._get_col(X, col, 
                self.config["data_specs"]["feature_methods"][col]))
            if isinstance(vec, csr_matrix):
                transformed.append(vec)
        return hstack(transformed)     

    def _get_col(self, X, col, method):
        """ Configures input data for vectorization """

        if method == "numeric":
            return [x[col] if ~pd.isnull(x[col]) else 0.0 for x in X]
        else:
            return [force_str(x[col]) for x in X]

class Normify(TransformerMixin, BaseEstimator):    
    """ Apply Normalize and/or Standardize transformers """

    def __init__(self, methods=[]):
        """ Set input methods as attribute """

        self.methods = methods        

    def fit(self, X, y=None):
        """ Fit desired methods """

        self.models = []
        for method in self.methods:
            proc = pre_processors[method]
            if method == "Standard Scaler":
                self.models.append(proc(with_mean=False))
            else:
                self.models.append(proc())
        if len(self.methods) == 0:
            pass
        elif len(self.methods) == 1:
            self.models[0].fit(X)
        elif len(self.methods) == 2:
            self.models[1].fit(self.models[0].fit_transform(X))
        else:
            raise Exception("Only 2 normalization methods allowed")
        return self

    def transform(self, X):
        """ Transform non-normal input to Normified output """

        if len(self.methods) == 0:
            return X
        elif len(self.methods) == 1:
            return self.models[0].transform(X)
        elif len(self.methods) == 2:
            return self.models[1].transform(self.models[0].transform(X))

class Sparsify(TransformerMixin, BaseEstimator):
    """ Transform dense input to sparse """

    def __init__(self, scale=False, strategy="median"):
        """ Set input arguments as attributes """

        self.scale = scale
        self.strategy = strategy

    def fit(self, X, y=None):
        """ Return instance """
        
        self.imputer = SimpleImputer(strategy=self.strategy)
        X_imp = self.imputer.fit_transform([[self._force_float(x)] for x in X])
        if self.scale:
            self.scaler = MinMaxScaler(feature_range=(1, 100)).fit(X_imp)    
        return self

    def transform(self, X):
        """ Dense input to sparse output """

        X_imp = self.imputer.transform([[self._force_float(x)] for x in X])
        if self.scale:
            return csr_matrix(self.scaler.transform(X_imp))
        else:
            return csr_matrix(X_imp)

    def _force_float(self, x):
        try:
            return float(x)
        except:
            return np.nan

class Densify(TransformerMixin, BaseEstimator):
    """ Transform sparse input to dense """
    def fit(self, X, y=None):
        """ Return instance """

        return self

    def transform(self, X):
        """ Sparse input to dense output """

        return X.toarray()

class Encoder(TransformerMixin):
    """ Label and onehot encoder pipeline"""

    def fit(self, X):
        """ Fit label encoder and onehot encoder"""

        self.label = LabelEncoder()
        self.onehot = OneHotEncoder(categories='auto')
        self.onehot.fit([[x] for x in self.label.fit_transform(X)])
        return self

    def transform(self, X):
        """ Transform using fitted label encoder and onehot encoder"""

        return self.onehot.transform([[x] for x in self.label.transform(X)])

class Vec_(TransformerMixin):
    """ Vectorizer wrapper with custom display """ 

    def __init__(self, tf):
        """ Set input vectorizer as attribute """

        self.tf = tf

    def transform(self, X):
        """ Transform using transformer attribute """

        return self.tf.transform(X)

    def __repr__(self):
        """ Representation """

        return "Vec"

    def __str__(self):
        """ Print string """

        return "Vec"

###########################################
### Utility Functions for Classificator ###
###########################################

def force_str(x):
    """ Force string """

    try:
        return str(x)
    except:
        try:
            return "".join(i for i in x if ord(i)<128)
        except:
            return ""

def load_sklearn_dataset(name):
    """ Load dataset from sklearn and return as pandas dataframe """

    if name in ["iris"]:
        data = datasets[name]() 
        df = (
            pd.DataFrame(
                np.hstack([data.data, [[x] for x in data.target]]), 
                columns=(list(data.feature_names) + ["label"])))
    elif name in ["newsgroups"]:
        data = datasets[name](subset='train', remove=('headers', 'footers', 'quotes'))
        df = pd.DataFrame(
            np.hstack([[[x] for x in data.data], [[x] for x in data.target]]), 
            columns=["text", "label"]).sample(1000, random_state=10) 
    else:
        raise Exception("Unknown dataset")
    return df

def is_number(s):
    """ Is input string a number """

    try:
        dummy = float(s)
        return True
    except ValueError:
        return False

def classifaction_report_df(report):
    """ Convert classification report to pandas dataframe """

    report_data = []
    lines = report.split('\n')
    for line in lines:
        try:
            row = {}
            row_data = line.strip().split()
            for index in range(len(row_data)):
                if is_number(row_data[index]):
                    start_ind = index
                    break
            row['class'] = " ".join(row_data[:(start_ind)])
            row['precision'] = float(row_data[start_ind])
            row['recall'] = float(row_data[start_ind + 1])
            row['f1_score'] = float(row_data[start_ind + 2])
            row['support'] = float(row_data[start_ind + 3])
            report_data.append(row)
        except:
            pass
    out = pd.DataFrame.from_dict(report_data)
    out["support"] = out["support"].apply(int)
    return out

