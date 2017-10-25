import json
from collections import Counter
from multiprocessing import Process

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier as clf

from eda.core import load_population
from eda.dataset import load_sets
from core import check_distribution, __encode_gm__
from datetime import datetime as dt

from eda.integration import __ensemble_predict__


def __save_population__(population, fitness):
    print '\tsaving population...'

    medians = np.median(fitness, axis=0)
    selected = np.multiply.reduce(fitness >= medians, axis=1)
    ref = np.flatnonzero(selected)[0]

    pd.DataFrame(population[ref]).to_csv('selection_population.csv', sep=',', index=False, header=False)


def distinct_failure_diversity(predictions, y_true):
    """
    Imlements distinct failure diversity. See
        Derek Partridge & Wo jtek Krzanowski. Distinct Failure Diversity in Multiversion Software. 1997
        for more information.

    :param predictions:
    :param y_true:
    :return:
    """
    n_classifiers, n_instances = predictions.shape
    distinct_failures = np.zeros(n_classifiers, dtype=np.float32)

    for i in xrange(n_instances):
        truth = y_true[i]
        count = Counter(predictions[:, i])
        for cls, n_votes in count.items():
            if cls != truth:
                distinct_failures[n_votes - 1] += 1

    distinct_failures_count = np.sum(distinct_failures)

    dfd = 0.
    for j in xrange(n_classifiers):
        dfd += (float(n_classifiers - (j + 1))/float(n_classifiers - 1)) * (distinct_failures[j] / distinct_failures_count)

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


def eda_select(gen_pop, val_predictions, y_val, n_individuals=100, n_generations=100, test_predictions=None, save_every=5):
    n_classifiers, n_features = gen_pop.shape
    n_objectives = 2

    gm = np.full(n_classifiers, 0.5, dtype=np.float32)
    sel_pop = np.empty((n_individuals, n_classifiers), dtype=np.bool)
    fitness = np.empty((n_individuals, n_objectives), dtype=np.float32)

    t1 = dt.now()

    for g in xrange(n_generations):
        for i in xrange(n_individuals):
            for j in xrange(n_classifiers):
                sel_pop[i, j] = np.random.choice(a=[True, False], p=[gm[j], 1. - gm[j]])

            fitness[i, :] = get_selection_fitness(val_predictions[sel_pop[i]], y_val)

        medians = np.median(fitness, axis=0)
        means = np.mean(fitness, axis=0)

        t2 = dt.now()

        print 'generation %2.d: median: (%.4f, %.4f) mean: (%.4f, %.4f) time elapsed: %f' % (
            g, medians[0], medians[1], means[0], means[1], (t2 - t1).total_seconds()
        )
        t1 = t2

        if (g % save_every == 0) and (g > 0):
            Process(
                target=__save_population__,
                kwargs=dict(population=sel_pop, fitness=fitness)
            ).start()

        gm = __encode_gm__(sel_pop, fitness)

    medians = np.median(fitness, axis=0)
    selected = np.multiply.reduce(fitness >= medians, axis=1)

    Process(
        target=__save_population__,
        kwargs=dict(population=sel_pop, fitness=fitness)
    ).start()

    return sel_pop[np.flatnonzero(selected)[0]]


def main():
    params = json.load(open('../params.json', 'r'))

    print 'loading datasets...'

    X_train, X_val, X_test, y_train, y_val, y_test = load_sets(
        params['train_path'],
        params['val_path'],
        params['test_path']
    )

    print 'loading population...'
    _population = pd.read_csv('generation_population.csv', sep=',').values

    _ensemble, _population, val_predictions = load_population(clf, _population, X_train, y_train, X_val, y_val)
    # _, _, test_predictions = load_population(clf, _population, X_train, y_train, X_test, y_test)

    _best_classifiers = eda_select(
        _population, val_predictions, y_val,
        n_individuals=500,
        n_generations=10,
    )

    dummy_weights = np.ones((len(_best_classifiers), len(np.unique(y_val))), dtype=np.float32)

    ensemble_preds = __ensemble_predict__(dummy_weights, val_predictions[_best_classifiers])
    ensemble_acc = accuracy_score(y_val, ensemble_preds)

    print 'ensemble validation accuracy: %.2f' % ensemble_acc


if __name__ == '__main__':
    main()
