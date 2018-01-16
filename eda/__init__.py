import csv
import datetime
import itertools as it
import operator as op
import os
from collections import Counter
from datetime import datetime as dt
from multiprocessing import Process, Manager, Lock

import numpy as np
import pandas as pd
from pathlib2 import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier

from utils import flatten
from sklearn.ensemble import AdaBoostClassifier


class Ensemble(object):
    def __init__(
            self,
            X_train, y_train,
            n_classifiers=None, classifiers=None, base_classifier=None,
            n_features=None, features=None,
            activated=None, voting_weights=None,
    ):
        """

        :type X_train: pandas.DataFrame
        :param X_train:
        :param y_train:
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
            self.classifiers = [DummyClassifier() for x in xrange(n_classifiers)]
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
    def from_adaboost(cls, X_train, y_train, n_classifiers, n_generations, use_weights=False):

        rf = AdaBoostClassifier(n_estimators=n_classifiers, algorithm='SAMME')  # type: AdaBoostClassifier
        rf = rf.fit(X_train, y_train)

        n_classes = rf.n_classes_

        voting_weights = np.empty((n_classifiers, n_classes), dtype=np.float32)

        if use_weights:
            voting_weights[:] = rf.estimator_weights_
            voting_weights = voting_weights.T
        else:
            voting_weights[:] = 1

        ensemble = Ensemble(
            X_train=X_train, y_train=y_train,
            classifiers=rf.estimators_,
            features=np.ones(
                (n_classifiers, X_train.shape[1]), dtype=np.int32
            ),
            activated=np.ones(n_classifiers, dtype=np.int32),
            voting_weights=voting_weights
        )
        return ensemble


    @property
    def fitness(self):
        raise NotImplementedError('not implemented yet!')

    def get_genotype(self, index):
        return self.truth_features[index, :]

    def set_classifier(self, index, model, truth_features):
        assert index < self.n_classifiers, \
            ValueError('index must be a value lesser than the number of total classifiers!')

        selected_features = self.feature_names[truth_features]

        self.classifiers[index] = model

        self.truth_features[index, :] = truth_features
        self.train_preds[index, :] = model.predict(self.X_train[selected_features])
        self.val_preds[index, :] = model.predict(self.X_val[selected_features])

        return self

    def train_classifier_with_features(self, index, base_classifier, feature_index):
        """

        :param index: classifier to be replaced.
        :param base_classifier: Type of the base classifier.
        :param feature_index: A boolean array denoting the selected features for this classifier.
        """

        assert index < self.n_classifiers, \
            ValueError('index must be a value lesser than the number of total classifiers!')

        model = base_classifier(random_state=0, max_depth=5)  # type: DecisionTreeClassifier

        selected_features = self.feature_names[feature_index]

        if np.max(feature_index) == 0:
            model = DummyClassifier()
        else:
            model = model.fit(self.X_train[selected_features], self.y_train)

        self.classifiers[index] = model

        self.truth_features[index, :] = feature_index
        self.train_preds[index, :] = model.predict(self.X_train[selected_features])
        self.val_preds[index, :] = model.predict(self.X_val[selected_features])

        return self


    @classmethod
    def create_base(cls, X_train, y_train, X_val, y_val, base_classifier, n_classifiers, n_features):
        base = cls(
            X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val,
            base_classifier=base_classifier, n_classifiers=n_classifiers, n_features=n_features,
        )

        return base

    @classmethod
    def load_population(cls, base_classifier, population, X_train, y_train, X_val, y_val, verbose=True):
        raise NotImplementedError('not implemented yet!')

        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val

        X_features = X_train.columns

        n_classifiers, n_attributes = population.shape

        ensemble = np.empty(n_classifiers, dtype=np.object)
        predictions = np.empty((n_classifiers, X_val.shape[0]), dtype=np.int32)

        for j in xrange(n_classifiers):  # number of base classifiers
            selected_features = X_features[np.flatnonzero(population[j])]
            ensemble[j], predictions[j] = __get_classifier__(
                base_classifier, selected_features,
                X_train, y_train, X_val
            )
            if j % 50 == 0 and verbose:
                print 'Loaded %d classifiers' % j

        return ensemble, population, predictions

    def get_predictions(self, X):
        """
        Given a list of classifiers and the features each one of them uses, returns a matrix of predictions for dataset X.

        :param X: A dataset comprised of instances and attributes.
        :param preds: optional - matrix where each row is a classifier and each column an instance.
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

    def predict_prob(self, X):
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

        :param X: A dataset comprised of instances and attributes.
        :param preds: optional - matrix where each row is a classifier and each column an instance.
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
        preds = self.get_predictions(X)
        _dfd = self.distinct_failure_diversity(preds, y)
        return _dfd

    @staticmethod
    def distinct_failure_diversity(predictions, y_true):
        """
        Implements distinct failure diversity. See
            Derek Partridge & Wo jtek Krzanowski. Distinct Failure Diversity in Multiversion Software. 1997
            for more information.

        :type predictions: numpy.ndarray
        :param predictions:
        :type y_true: pandas.Series
        :param y_true:
        :return:
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


class Reporter(object):
    metrics = [
        ('accuracy', accuracy_score),
        ('precision-micro', lambda y_true, y_pred: precision_score(y_true, y_pred, average='micro')),
        ('precision-macro', lambda y_true, y_pred: precision_score(y_true, y_pred, average='macro')),
        ('precision-weighted', lambda y_true, y_pred: precision_score(y_true, y_pred, average='weighted')),
        # ('precision-samples', lambda y_true, y_pred: precision_score(y_true, y_pred, average='samples')),
        ('recall-micro', lambda y_true, y_pred: recall_score(y_true, y_pred, average='micro')),
        ('recall-macro', lambda y_true, y_pred: recall_score(y_true, y_pred, average='macro')),
        ('recall-weighted', lambda y_true, y_pred: recall_score(y_true, y_pred, average='weighted')),
        # ('recall-samples', lambda y_true, y_pred: recall_score(y_true, y_pred, average='samples')),
        ('f1-micro', lambda y_true, y_pred: f1_score(y_true, y_pred, average='micro')),
        ('f1-macro', lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro')),
        ('f1-weighted', lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted')),
        # ('f1-samples', lambda y_true, y_pred: f1_score(y_true, y_pred, average='samples')),
    ]

    def __init__(self, Xs, ys, n_classifiers, n_classes, set_names, dataset_name, n_fold, n_run, output_path, alias=None, n_jobs=4):
        self.Xs = Xs
        self.ys = ys
        self.set_sizes = map(len, self.ys)
        self.set_names = set_names
        self.dataset_name = dataset_name
        if alias is None:
            self.alias = str(dt.now())
        else:
            if isinstance(alias, datetime.datetime):
                alias = str(alias)
            self.alias = alias
        self.n_run = n_run
        self.n_fold = n_fold
        self.n_classifiers = n_classifiers
        self.n_classes = n_classes
        self.output_path = output_path

        self.population_file = os.path.join(
            self.output_path, '-'.join([self.alias, self.dataset_name, str(self.n_fold), str(self.n_run), 'pop']) + '.csv'
        )
        with open(self.population_file, 'wb') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(
                ['dataset', 'n_fold', 'n_run', 'generation', 'set_name', 'elite', 'fitness'] +
                [a for a, b in Reporter.metrics] +
                ['w_%d_%d' % (a, b) for a, b in it.product(np.arange(self.n_classifiers), np.arange(self.n_classes))]
            )

        self.gm_file = os.path.join(
            self.output_path,
            '-'.join([self.alias, self.dataset_name, str(self.n_fold), str(self.n_run), 'gm']) + '.csv'
        )
        with open(self.gm_file, 'wb') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(
                ['dataset', 'n_fold', 'n_run', 'generation', 'scale'] +
                ['w_%d_%d' % (a, b) for a, b in it.product(np.arange(self.n_classifiers), np.arange(self.n_classes))]
            )

    # def __get_hash__(self, func):
    #     return hash(func.__name__ + str(self.n_fold))

    def save_population(self, generation, elite, ensembles, P_fitness):

        with open(self.population_file, 'ab') as f:
            writer = csv.writer(f, delimiter=',')

            counter = 0
            for elite, ensemble, fitness in it.izip(elite, ensembles, P_fitness):
                ravel_weights = ensemble.voting_weights.ravel().tolist()

                for set_name, set_x, set_y in it.izip(self.set_names, self.Xs, self.ys):
                    preds = ensemble.predict(set_x)
                    results = []
                    for metric_name, metric_func in Reporter.metrics:
                        results += [metric_func(y_true=set_y, y_pred=preds)]

                    writer.writerow(
                        [self.dataset_name, self.n_fold, self.n_run, generation, set_name, elite, fitness] + results + ravel_weights
                    )
                    counter += 1

    def save_gm(self, generation, loc, scale):
        with open(self.gm_file, 'ab') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(
                [self.dataset_name, self.n_fold, self.n_run, generation, scale] +
                loc.ravel().tolist()
            )

# class DummyIterator(object):
#     def __init__(self, value, length, reset=True):
#         self.current = 0
#         self.length = length
#         self.value = value
#         self.reset = reset
#
#     def __len__(self):
#         return self.length
#
#     def __iter__(self):
#         return self
#
#     def next(self):
#         if self.current < self.length:
#             self.current += 1
#             return self.value
#         else:
#             if self.reset:
#                 self.current = 0
#             raise StopIteration
#
#
class DummyClassifier(object):
    def __init__(self):
        pass

    def fit(self, X_train, y_train):
        pass

    def predict(self, X_test):
        return np.ones(X_test.shape[0], dtype=np.int32) * -1

# ---------------------------------------------------#
# ----------- # pareto-related methods # ----------- #
# ---------------------------------------------------#

def pairwise_domination(P, Q=None):
    """

    :type P: numpy.ndarray
    :param P: Population of individuals. If Q is not provided, it will calculate the pairwise domination among
        P individuals.
    :type Q: numpy.ndarray
    :param Q: optional - second population of individuals. If provided, will calculate domination among P and Q
        populations, in a way that in the cell D[i, j], for example, denotes the dominance of the i-th individual
        of P over the j-th individual of Q.
    :rtype: numpy.ndarray
    :return: An matrix where each cell determines if the individual in that row
        dominates the individual in that column, and vice-versa.
    """

    n_p_individuals, = P.shape
    n_q_individuals, = Q.shape if Q is not None else P.shape

    matrix = np.empty((n_p_individuals, n_q_individuals), dtype=np.int32)
    for i in xrange(n_p_individuals):
        for j in xrange(n_q_individuals):
            matrix[i, j] = a_dominates_b(P[i], P[j])

    return matrix


def a_dominates_b(a, b):
    """

    :param a: list of performance in the n objective functions of individual a
    :param b: list of performance in the n objective functions of individual b
    :return: -1 if b dominates a, +1 if the opposite, and 0 if there is no dominance (i.e. same front)
    """
    assert type(a) == type(b), TypeError('a and b must have the same type!')
    assert type(a) in [np.void, np.ndarray, list], TypeError(
        'invalid type for a and b! Must be either lists or void objects!')

    if isinstance(a, np.void):
        newa = [a[name] for name in a.dtype.names]
        newb = [b[name] for name in b.dtype.names]
    else:
        newa = a
        newb = b

    a_dominates = np.any(newa > newb) and np.all(newa >= newb)
    b_dominates = np.any(newb > newa) and np.all(newb >= newa)

    res = (a_dominates * 1) + (b_dominates * -1)
    return res


def crowding_distance_sort(P, indices):
    """
        Worst case scenario for this function: O(m * P * log(P)), where m
        is the number of objectives and P the size of population.

        Adapted from A fast elitist non-dominated sorting genetic algorithm for multi-objective optimization: NSGA-II

        :type P: numpy.ndarray
        :param P: An numpy.ndarray with m dimensions,
            where the first value in the tuple is the quality of the solution
            in the first objective, and so on and so forth.
        :type indices: list
        :param indices: indices for individuals in the current front.
        :rtype: numpy.ndarray
        :return: A list of crowding distances.
        """

    n_individuals, n_objectives = P[indices].shape
    redux = np.empty((n_individuals, n_objectives + 2), dtype=np.float32)  # for index and crowding distance
    redux[:, 0] = indices  # indices
    redux[:, 1] = 0.  # crowding distance starts at zero
    redux[:, [2, 3]] = P[indices]  # absorbs quality of individuals

    for objective in xrange(2, 2 + n_objectives):
        redux = redux[redux[:, objective].argsort()]
        redux[[0, -1], 1] = np.inf  # edge individuals have maximum crowding distance
        for i in xrange(1, n_individuals - 1):  # from 1-th individual to N-1 (included)
            redux[i, 1] = redux[i, 1] + abs(redux[i + 1, objective] - redux[i - 1, objective])

    redux = redux[redux[:, 1].argsort()[::-1]]
    return redux[:, 0].astype(np.int32)


def get_fronts(pop):
    """
    Based on a population of individuals, order them based on their Pareto dominance. Adapted from
    A fast elitist non-dominated sorting genetic algorithm for multi-objective optimization: NSGA-II.

    :param pop: A matrix where rows are individuals and columns their fitness in each one of the objectives.
    :type pop: numpy.ndarray
    :return:
    """

    n_individuals, n_objectives = pop.shape

    added = np.zeros(n_individuals, dtype=np.bool)
    count_dominated = np.zeros(n_individuals, dtype=np.int32)
    list_dominates = [[] for x in xrange(n_individuals)]
    fronts = []

    for i in xrange(n_individuals):
        for j in xrange(i + 1, n_individuals):
            res = a_dominates_b(pop[i], pop[j])
            if res == 1:
                count_dominated[j] += 1
                list_dominates[i] += [j]
            elif res == -1:
                count_dominated[i] += 1
                list_dominates[j] += [i]

    # iteratively process fronts
    n_front = 0
    while sum(added) < n_individuals:
        _where = (count_dominated == 0)

        # there are only individuals that do not dominate any other remaining; add those to the last 'front'
        if _where.max() == 0:
            current_front = np.flatnonzero(added == 0).tolist()
            fronts += [crowding_distance_sort(pop, current_front)]
            break

        added[_where] = True  # add those individuals to the 'processed' list

        current_front = np.flatnonzero(_where)

        fronts += [crowding_distance_sort(pop, current_front)]
        count_dominated[current_front] = -1

        _chain = set(reduce(
            op.add,
            map(
                lambda (i, x): list_dominates[i],
                enumerate(_where))
        )
        )
        count_dominated[list(_chain)] -= 1  # decrements one domination

        n_front += 1

    return fronts
