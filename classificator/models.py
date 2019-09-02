"""
Model dictionaries
"""

import sklearn.ensemble
import sklearn.tree
import sklearn.linear_model
import sklearn.neural_network
import sklearn.naive_bayes
import sklearn.feature_extraction
import sklearn.preprocessing
import sklearn.feature_selection
import sklearn.svm
import sklearn.metrics
import sklearn.neighbors

from sklearn.datasets import load_iris, fetch_20newsgroups

from .combined import CombinedEstimator

__all__ = [
    "classifiers", 
    "vectorizers", 
    "pre_processors", 
    "feature_selectors", 
    "scores", 
    "datasets", 
    "require_dense"
    ]

classifiers = {
    "Random Forest": sklearn.ensemble.RandomForestClassifier,
    "Decision Tree": sklearn.tree.DecisionTreeClassifier,
    "Extra Trees": sklearn.ensemble.ExtraTreesClassifier,
    "GB Trees": sklearn.ensemble.GradientBoostingClassifier,
    "Logistic Regression": sklearn.linear_model.LogisticRegression,
    "Passive Aggressive": sklearn.linear_model.PassiveAggressiveClassifier,
    "Ridge": sklearn.linear_model.RidgeClassifier,
    "Stochastic Gradient Descent": sklearn.linear_model.SGDClassifier,
    "Multi-layered Perceptron": sklearn.neural_network.MLPClassifier,
    "Gaussian Naive Bayes": sklearn.naive_bayes.GaussianNB,
    "Multinomial Naive Bayes": sklearn.naive_bayes.MultinomialNB,
    "Bernoulli Naive Bayes": sklearn.naive_bayes.BernoulliNB,
    "Support Vector Machine": sklearn.svm.SVC,
    "Linear SVM": sklearn.svm.LinearSVC,
    "KNN": sklearn.neighbors.KNeighborsClassifier,
    "Nearest Centroid": sklearn.neighbors.NearestCentroid,
    "Combined Estimator": CombinedEstimator
}

vectorizers = {
     "Count Vectorizer": sklearn.feature_extraction.text.CountVectorizer,
     "TF-IDF Vectorizer": sklearn.feature_extraction.text.TfidfVectorizer,
     "Hashing Vectorizer": sklearn.feature_extraction.text.HashingVectorizer,
}

pre_processors = {
    "Standard Scaler": sklearn.preprocessing.StandardScaler,
    "Max Scaler": sklearn.preprocessing.MaxAbsScaler,
    "Normalizer": sklearn.preprocessing.Normalizer
}

feature_selectors = {
    "Chi-Squared": sklearn.feature_selection.chi2
}

scores = {
    "Accuracy": sklearn.metrics.make_scorer(sklearn.metrics.accuracy_score),
    "F1 (binary)": sklearn.metrics.make_scorer(sklearn.metrics.f1_score),
    "F1 (macro)": sklearn.metrics.make_scorer(sklearn.metrics.f1_score, average="macro"),
    "F1 (weighted)": sklearn.metrics.make_scorer(sklearn.metrics.f1_score, average="weighted"),
    "AUC ROC": sklearn.metrics.make_scorer(sklearn.metrics.roc_auc_score)
}

# Sklearn data sets
datasets = {
    "iris": load_iris,
    "newsgroups": fetch_20newsgroups
}

# List model keys that require densification before fitting
require_dense = ["GB Trees", "Gaussian Naive Bayes"]
