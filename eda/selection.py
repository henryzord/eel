import json
from collections import Counter
from multiprocessing import Process

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier as clf

from eda.core import load_population, get_classes, DummyIterator
from core import check_distribution, __pareto_encode_gm__
from datetime import datetime as dt


def distinct_failure_diversity(predictions, y_true):
    """
    Imlements distinct failure diversity. See
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

    if distinct_failures_count > 0:
        for j in xrange(1, n_classifiers + 1):
            dfd += (float(n_classifiers - j)/float(n_classifiers - 1)) * \
                   (float(distinct_failures[j]) / distinct_failures_count)

    return dfd


def simple_select(ensemble, population, predictions, X_val, y_val):
    accs = np.sum(predictions, axis=1) / float(predictions.shape[1])

    median = np.median(accs)
    # # warnings.warn('warning: using whole population')
    to_select = np.ones(len(population), dtype=np.bool)
    # to_select = accs > median

    best_ensemble = ensemble[to_select]
    best_population = population[to_select]
    predictions = predictions[to_select]

    check_distribution(best_ensemble, best_population, X_val, y_val)

    return best_ensemble, best_population, predictions


def get_selection_fitness(individual_preds, y_true):
    n_classifiers, n_instances = individual_preds.shape

    counts = map(lambda x: Counter(individual_preds[:, x]).most_common(1)[0][0], xrange(n_instances))
    acc = accuracy_score(y_true, counts)
    div = distinct_failure_diversity(individual_preds, y_true)

    return acc, div


def eda_select(
        features, classifiers, val_predictions, y_val,
        n_individuals=100, n_generations=100, save_every=5, reporter=None
):
    """


    :param features:
    :param val_predictions:
    :param y_val:
    :param n_individuals:
    :param n_generations:
    :param save_every:
    :type reporter: eda.Reporter
    :param reporter:
    :return:
    """

    n_classifiers, n_features = features.shape
    n_objectives = 2

    gm = np.full(n_classifiers, 0.5, dtype=np.float32)
    sel_pop = np.empty((n_individuals, n_classifiers), dtype=np.bool)
    fitness = np.empty((n_individuals, n_objectives), dtype=np.float32)

    n_classes = len(np.unique(y_val))

    dummy_weights = np.empty(
        (n_individuals, n_classifiers, n_classes),
        dtype=np.float32
    )

    t1 = dt.now()

    for g in xrange(n_generations):
        for i in xrange(n_individuals):
            for j in xrange(n_classifiers):
                sel_pop[i, j] = np.random.choice(a=[True, False], p=[gm[j], 1. - gm[j]])
                dummy_weights[i, j, :] = sel_pop[i, j]

            fitness[i, :] = get_selection_fitness(val_predictions[sel_pop[i]], y_val)

        medians = np.median(fitness, axis=0)
        means = np.mean(fitness, axis=0)

        t2 = dt.now()

        reporter.callback(eda_select, g, dummy_weights, features, classifiers)
        reporter.save_population(eda_select, sel_pop, g, save_every)

        print 'generation %2.d: median: (%.4f, %.4f) mean: (%.4f, %.4f) time elapsed: %f' % (
            g, medians[0], medians[1], means[0], means[1], (t2 - t1).total_seconds()
        )
        t1 = t2

        gm = __pareto_encode_gm__(sel_pop, fitness)

    medians = np.median(fitness, axis=0)
    selected = np.multiply.reduce(fitness >= medians, axis=1)

    reporter.save_population(eda_select, sel_pop)

    return selected
