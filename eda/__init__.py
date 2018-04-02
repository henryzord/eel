from collections import Counter

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier


class Ensemble(object):
    def __init__(
            self,
            X_train, y_train,
            n_classifiers=None, classifiers=None, base_classifier=None,
            n_features=None, features=None,
            activated=None, voting_weights=None
    ):
        """
        Builds an ensemble of classifiers.

        :type X_train: pandas.DataFrame
        :param X_train: Predictive attributes of the training instances.
        :param y_train: Labels of training instances.
        :param classifiers: A list of classifiers that are in compliance
            with the (fit, predict) interface of sklearn classifiers.
        :param features: An matrix where each row is a classifier and each
            column denotes the absence or presence of that attribute for the given classifier.
        :param activated: A boolean array denoting the activated classifiers.
        :param voting_weights: An matrix of shape (n_classifiers, n_classes).
        """

        assert isinstance(X_train, pd.DataFrame), TypeError('X_train must be a pandas.DataFrame!')

        scratch = ((n_classifiers is not None) and (base_classifier is not None))

        if scratch:
            if X_train is None or y_train is None:
                raise ValueError(
                    'When building an ensemble from scratch, training and validation sets must be provided!'
                )

            self.base_classifier = base_classifier
            self.classifiers = []
            for i in xrange(n_classifiers):
                self.classifiers += [DummyClassifier()]
        elif classifiers is not None:
            self.base_classifier = type(classifiers[0])
            self.classifiers = classifiers
        else:
            raise ValueError(
                'Either a list of classifiers or the number of '
                'base classifiers (along with the base type) must be provided!'
            )
        # finally:
        self.n_classifiers = len(self.classifiers)

        if features is not None:
            self.truth_features = features
        elif n_features is not None:
            self.truth_features = np.zeros(
                (self.n_classifiers, n_features), dtype=np.int32
            )
        else:
            raise ValueError('Either a list of activated features or the number of features must be provided!')
        # finally:
        self.n_features = len(self.truth_features[0])

        self.X_train = X_train
        self.y_train = y_train
        self.feature_names = self.X_train.columns

        self.n_classes = len(np.unique(y_train))

        if voting_weights is not None:
            self.voting_weights = voting_weights
        else:
            self.voting_weights = np.ones(self.n_classifiers, dtype=np.float32)

        if activated is not None:
            self.activated = activated
        else:
            self.activated = np.ones(self.n_classifiers, dtype=np.int32)

        n_instances_train = self.X_train.shape[0]

        self.train_preds = np.empty((self.n_classifiers, n_instances_train), dtype=np.int32)

    @classmethod
    def from_adaboost(cls, X_train, y_train, n_classifiers):
        """
        Initializes the ensemble using AdaBoost as generator of the base classifiers, as well as the voting weights.

        :type X_train: pandas.DataFrame
        :param X_train: Predictive attributes of the training instances.
        :param y_train: Labels of training instances.
        :param n_classifiers: Number of base classifiers to use within AdaBoost.
        :return: an ensemble of base classifiers trained by AdaBoost.
        """

        rf = AdaBoostClassifier(n_estimators=n_classifiers, algorithm='SAMME')  # type: AdaBoostClassifier
        rf = rf.fit(X_train, y_train)  # type: AdaBoostClassifier

        n_classes = rf.n_classes_

        voting_weights = np.empty((n_classifiers, n_classes), dtype=np.float32)
        voting_weights[:] = rf.estimator_weights_[:, np.newaxis]

        ensemble = Ensemble(
            X_train=X_train, y_train=y_train,
            classifiers=rf.estimators_,
            features=np.ones(
                (n_classifiers, X_train.shape[1]), dtype=np.int32
            ),
            activated=np.ones(n_classifiers, dtype=np.int32),
            voting_weights=voting_weights,
        )
        return ensemble

    def resample_voting_weights(self, loc, scale):
        """
        Samples voting weights for the whole ensemble from a normal distribution.

        :type loc: float
        :param loc: the mean of the normal distribution.
        :type scale: float
        :param scale: the std deviation of the normal distribution.
        :rtype: Ensemble
        :return: returns self.
        """

        for j in xrange(self.n_classifiers):
            for c in xrange(self.n_classes):
                self.voting_weights[j][c] = np.clip(
                    np.random.normal(loc=loc[j][c], scale=scale),
                    a_min=0., a_max=1.
                )

        return self

    def get_predictions(self, X):
        """
        Given a list of classifiers and the features each one of them uses,
        returns a matrix of predictions for dataset X.

        :param X: A dataset comprised of instances and attributes.
        :return: An matrix where each row is a classifier and each column an instance in X.
        """

        X_features = X.columns

        n_activated = np.count_nonzero(self.activated)
        index_activated = np.flatnonzero(self.activated)

        if X is self.X_train:
            preds = self.train_preds
        else:
            preds = np.empty((n_activated, X.shape[0]), dtype=np.int32)

        for raw, j in enumerate(index_activated):  # number of base classifiers
            selected_features = X_features[np.flatnonzero(self.truth_features[j])]
            preds[raw, :] = self.classifiers[j].predict(X[selected_features])

        return preds

    def predict_proba(self, X):
        """
        Predict probabilities for the given instances.

        :param X: A dataset comprised of instances and attributes.
        :return: A matrix where each row is an instance and each column the probability for that instance being
            labeled to that class.
        """

        preds = self.get_predictions(X)

        n_classifiers, n_instances = preds.shape

        global_votes = np.zeros((n_instances, self.n_classes), dtype=np.float32)

        for i in xrange(n_instances):
            for j in xrange(n_classifiers):
                global_votes[i, preds[j, i]] += self.voting_weights[j, preds[j, i]]

        _sum = np.sum(global_votes, axis=1)

        return global_votes / _sum[:, None]

    def predict(self, X):
        """
        Predicts the class of a set of instances.

        :param X: A dataset comprised of instances and attributes.
        :return: An array where each position contains the ensemble prediction for that instance.
        """
        preds = self.get_predictions(X)

        n_classifiers, n_instances = preds.shape

        local_votes = np.empty(self.n_classes, dtype=np.float32)
        global_votes = np.empty(n_instances, dtype=np.int32)

        for i in xrange(n_instances):
            local_votes[:] = 0.

            for j in xrange(n_classifiers):
                local_votes[preds[j, i]] += self.voting_weights[j, preds[j, i]]

            global_votes[i] = np.argmax(local_votes)

        return global_votes

    def dfd(self, X, y):
        """
        Calculates the distinct failure diversity of the ensemble within this instance.
        The measure is featured in the paper

        Partridge, Derek, and Wojtek Krzanowski. "Distinct failure diversity in multiversion software."
         Res. Rep 348 (1997): 24.

        :param X: Predictive attributes of the training instances.
        :param y: Labels of training instances.
        :return: the distinct failure diversity measure.
        """

        preds = self.get_predictions(X)
        _dfd = self.distinct_failure_diversity(preds, y)
        return _dfd

    @staticmethod
    def distinct_failure_diversity(predictions, y_true):
        """
        Implements the distinct failure diversity metric. Refer to

        Partridge, Derek, and Wojtek Krzanowski. "Distinct failure diversity in multiversion software."
         Res. Rep 348 (1997): 24.

        for more information.

        :type predictions: numpy.ndarray
        :param predictions: Predictions of the ensemble for a given set of instances.
        :type y_true: pandas.Series
        :param y_true: The ground truth labels of the predicted instances.
        :return: The distinct failure diversity measure.
        """

        if isinstance(predictions, pd.DataFrame):
            predictions = predictions.values

        if isinstance(y_true, pd.Series):
            y_true = y_true.tolist()

        n_classifiers, n_instances = predictions.shape
        distinct_failures = np.zeros(n_classifiers + 1, dtype=np.float32)

        for i in xrange(n_instances):
            truth = y_true[i]
            count = Counter(predictions[:, i])
            for cls, n_votes in count.items():
                if cls != truth:
                    distinct_failures[n_votes] += 1

        distinct_failures_count = np.sum(distinct_failures)  # type: int

        dfd = 0.

        if (distinct_failures_count > 0) and (n_classifiers > 1):
            for j in xrange(1, n_classifiers + 1):
                dfd += (float(n_classifiers - j) / float(n_classifiers - 1)) * \
                       (float(distinct_failures[j]) / distinct_failures_count)

        return dfd


class DummyClassifier(object):
    """
    Implements a DummyClassifier for compliance with the Ensemble class, when the base classifiers are not given
    to the constructor.
    """

    def fit(self, X_train, y_train):
        """
        Will not fit data.
        :param X_train: Predictive attributes of the training instances.
        :param y_train: Labels of training instances.
        :return: Returns self.
        """
        return self

    def predict(self, X_test):
        """
        Always predicts -1 for all instances.
        :param X_test: Instances for which to make predictions.
        :return: A list of labels of the same size of the input instances, but with only -1 values.
        """
        return np.ones(X_test.shape[0], dtype=np.int32) * -1
