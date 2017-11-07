import json
from datetime import datetime as dt

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier as clf

from eda.core import load_population, get_classes


def __ensemble_predict_matrix__(voting_weights, predictions):
    n_classifiers, n_classes = voting_weights.shape
    n_classifiers, n_instances = predictions.shape

    global_votes = np.empty((n_instances, n_classes), dtype=np.float32)

    for i in xrange(n_instances):
        global_votes[i, :] = 0.

        for j in xrange(n_classifiers):
            global_votes[i, predictions[j, i]] += voting_weights[j, predictions[j, i]]

    return global_votes


# -------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------- #


def integrate(val_predictions, y_val, n_individuals=100, n_generations=100, test_predictions=None, y_test=None):
    classes = np.unique(y_val)
    n_classes = len(classes)

    n_objectives = 2

    if test_predictions is not None:
        best_test_acc = -1

    n_classifiers, n_val_instances = val_predictions.shape

    population = np.empty((n_individuals, n_classifiers, n_classes), dtype=np.float32)
    fitness = np.empty(n_individuals, dtype=np.float32)

    # TODO distribution must be uniform in first iteration!

    min_weight = 0.1
    max_weight = 0.9
    # std = ((1./12.) * (max_weight - min_weight) ** 2.) ** (1./2.)
    std = 1.

    loc = np.random.uniform(size=(n_classifiers, n_classes), low=1., high=10.)
    scale = np.full((n_classifiers, n_classes), std, dtype=np.float32)

    t1 = dt.now()

    for g in xrange(n_generations):
        for i in xrange(n_individuals):
            for j in xrange(n_classifiers):
                for c in xrange(n_classes):
                    population[i, j, c] = np.random.normal(loc=loc[j, c], scale=scale[j, c])

            population[i] = np.clip(population[i], a_min=0., a_max=1.)

            y_val_pred = get_classes(population[i], val_predictions)
            fitness[i] = accuracy_score(y_val, y_val_pred)

            if test_predictions is not None:
                y_test_pred = get_classes(population[i], test_predictions)
                test_acc = accuracy_score(y_test, y_test_pred)

                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    related_val_acc = fitness[i]

        # update
        median = np.median(fitness)
        selected = fitness > median

        if np.count_nonzero(selected) == 0:
            print 'no individual is better than median; aborting...'
            break

        fit = population[selected]

        loc = np.mean(fit, axis=0)
        # scale = np.std(fit, axis=0)

        t2 = dt.now()

        # report
        print 'generation %d: min: %.7f median: %.7f mean: %.7f max: %.7f time elapsed: %f' % (
            g, np.min(fitness), np.median(fitness), np.mean(fitness), np.max(fitness), (t2 - t1).total_seconds()
        ) + (' best_test: %.2f' % best_test_acc) if y_test is not None else ''

        t1 = t2

    median = np.median(fitness)
    selected = np.flatnonzero(fitness > median)
    try:
        best_weights = population[selected[0], :, :]
    except IndexError:
        selected = np.flatnonzero(fitness >= median)
        best_weights = population[selected[0], :, :]

    return best_weights
