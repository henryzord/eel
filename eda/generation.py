import warnings
from datetime import datetime as dt

import numpy as np
from bitarray import bitarray
from sklearn.metrics import accuracy_score

from core import __pareto_encode_gm__, get_classes, distinct_failure_diversity, get_fronts
from eda.core import __get_classifier__, DummyIterator

'''
Check

> Using Bayesian Networks for Selecting Classifiers in GP Ensembles

for a measure on diversity.
 
'''


class ConversorIterator(object):
    def __init__(self, population):
        self.current = 0
        self.length = len(population)
        self.population = population

    def __getitem__(self, item):
        return np.array(list(self.population[item]))

    def __iter__(self):
        return self

    def __len__(self):
        return self.length

    def next(self):
        if self.current < self.length:
            ind = np.array(list(self.population[self.current]))
            self.current += 1
            return ind
        else:
            raise StopIteration


def get_generation_fitness(ensemble, fitness, val_predictions, y_val):
    """

    First objective is accuracy. Second objective is double-fault.
    see 'Genetic Algorithms with diversity measures to build classifier systems' for references

    :type ensemble:
    :param ensemble: List of classifiers.
    :type fitness: numpy.ndarray
    :param fitness: matrix to store fitness values.
    :type val_predictions: numpy.ndarray
    :param val_predictions: matrix where each row is a classifier and each column a prediction for that instance.
    :type y_val: numpy.ndarray
    :param y_val: array with real class for validation set.
    :rtype: numpy.ndarray
    :return: Returns a tuple where the first item is the fitness in the first objective, and so on and so forth.
    """
    from sklearn.preprocessing import normalize

    n_classifiers = len(ensemble)

    n_instances_val = y_val.shape[0]

    pairwise_double_fault = np.empty((n_classifiers, n_classifiers), dtype=np.float32)

    for i in xrange(n_classifiers):
        fitness[i, 0] = accuracy_score(y_val, val_predictions[i, :])

        for j in xrange(i, n_classifiers):
            # index = np.sum(np.logical_or(
            index = np.sum(np.logical_xor(
                val_predictions[i] == y_val,
                val_predictions[j] == y_val
            )) / float(n_instances_val)

            pairwise_double_fault[i, j] = index
            pairwise_double_fault[j, i] = index

        warnings.warn('WARNING: using max instead of min!')
        fitness[i, 1] = np.median(pairwise_double_fault[i, :])

    fitness = normalize(fitness, axis=0, norm='max')  # TODO remove later!
    return fitness


def generate(
        X_train, y_train, X_val, y_val, base_classifier,
        n_classifiers=100, n_generations=100,
        selection_strength=0.5,
        save_every=5, reporter=None
):
    """

    :param X_train:
    :param y_train:
    :param X_val:
    :param y_val:
    :param base_classifier:
    :param n_classifiers:
    :param n_generations:
    :param save_every:
    :type reporter: eda.Reporter
    :param reporter:
    :return:
    """

    X_features = X_train.columns
    n_features = len(X_features)
    n_objectives = 2  # accuracy and diversity

    # -- dummy weights -- #
    n_classes = len(np.unique(y_val))
    dummy_weights = np.ones((n_classifiers, n_classes), dtype=np.float32)
    dummy_weight_vector = DummyIterator(  # whole population of classifiers equals to one ensemble
        dummy_weights, length=1, reset=True
    )
    # -- dummy weights -- #

    n_instances_val = X_val.shape[0]

    initial_prob = 0.5
    gm = np.full(shape=n_features, fill_value=initial_prob, dtype=np.float32)  # initial gm

    classifiers = np.empty(n_classifiers, dtype=np.object)

    # first column for accuracy, second for scalar double fault
    P_fitness = np.empty((n_classifiers, n_objectives), dtype=np.float32)
    val_preds = np.empty((n_classifiers, n_instances_val), dtype=np.int32)

    # population
    P = [bitarray(n_features) for i in xrange(n_classifiers)]
    A = np.zeros(n_classifiers, dtype=np.bool)

    t1 = dt.now()

    g = 0
    while g < n_generations:
        for j in xrange(n_classifiers):
            if not A[j]:
                for k in xrange(n_features):
                    P[j][k] = np.random.choice(a=[0, 1], p=[1. - gm[k], gm[k]])

            selected_features = X_features[list(P[j])]
            classifiers[j], val_preds[j] = __get_classifier__(
                base_classifier, selected_features, X_train, y_train, X_val
            )

        ensemble_preds = get_classes(dummy_weights, val_preds)
        dfd = distinct_failure_diversity(val_preds, y_val)
        ensemble_acc = accuracy_score(y_val, ensemble_preds)

        P_fitness = get_generation_fitness(classifiers, P_fitness, val_preds, y_val)
        medians = np.median(P_fitness, axis=0)
        means = np.mean(P_fitness, axis=0)

        gm, A = __pareto_encode_gm__(A, P, P_fitness, select_strength=selection_strength)

        t2 = dt.now()

        try:
            reporter.save_accuracy(generate, g, dummy_weight_vector, ConversorIterator(P), classifiers)
            reporter.save_population(generate, P, g, save_every)
            reporter.save_gm(generate, g, gm)
        except AttributeError:
            pass

        print 'generation %2.d: ens val acc: %.2f dfd: %.4f median: (%.4f, %.4f) mean: (%.4f, %.4f) time elapsed: %f' % (
            g, ensemble_acc, dfd, medians[0], medians[1], means[0], means[1], (t2 - t1).total_seconds()
        )
        t1 = t2
        g += 1

    try:
        reporter.save_population(generate, P)
    except AttributeError:
        pass

    features = np.array(map(lambda x: x.tolist(), P))
    return classifiers, features, P_fitness
