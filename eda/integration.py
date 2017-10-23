import json

import numpy as np
from sklearn.metrics import accuracy_score

from eda.core import load_population
from eda.dataset import load_sets
import pandas as pd

from eda.selection import select
from sklearn.tree import DecisionTreeClassifier as clf
from datetime import datetime as dt


def __check_accuracy__(weights, models, features, X_test, y_test):

    n_classifiers = weights.shape[0]
    n_test_instances = X_test.shape[0]

    X_features = X_test.columns

    predictions = np.empty((n_classifiers, n_test_instances), dtype=np.int32)

    for j in xrange(n_classifiers):
        pred = models[j].predict(X_test[X_features[features[j]]])
        predictions[j] = y_test == pred

    y_test_pred = __ensemble_predict__(weights, predictions)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    return test_accuracy


def __ensemble_predict__(voting_weights, predictions):
    # TODO wrong! how can it predict if it doesn't know the class??

    n_classifiers, n_classes = voting_weights.shape
    n_classifiers, n_instances = predictions.shape

    local_votes = np.empty(n_classes, dtype=np.float32)
    global_votes = np.empty(n_instances, dtype=np.int32)

    for i in xrange(n_instances):
        local_votes[:] = 0.

        for j in xrange(n_classifiers):
            local_votes[predictions[j, i]] += voting_weights[j, predictions[j, i]]

        global_votes[i] = np.argmax(local_votes)

    return global_votes


# -------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------- #


def integrate(predictions, y_val, n_individuals=100, n_generations=100, models=None, features=None, X_test=None, y_test=None):
    classes = np.unique(y_val)
    n_classes = len(classes)

    n_classifiers, n_instances = predictions.shape

    population = np.empty((n_individuals, n_classifiers, n_classes), dtype=np.float32)
    fitness = np.empty(n_individuals, dtype=np.float32)

    min_weight = 0.1
    max_weight = 0.9
    std = ((1./12.) * (max_weight - min_weight) ** 2.) ** (1./2.)

    loc = np.full((n_classifiers, n_classes), 0.5, dtype=np.float32)
    scale = np.full((n_classifiers, n_classes), std, dtype=np.float32)

    t1 = dt.now()

    for g in xrange(n_generations):
        for i in xrange(n_individuals):
            for j in xrange(n_classifiers):
                for c in xrange(n_classes):
                    population[i, j, c] = np.random.normal(loc=loc[j, c], scale=scale[j, c])

            population[i] = np.clip(population[i], a_min=0, a_max=1)

            y_pred = __ensemble_predict__(population[i], predictions)
            fitness[i] = accuracy_score(y_val, y_pred)

        # update
        median = np.median(fitness)
        selected = fitness > median

        if np.count_nonzero(selected) == 0:
            print 'no individual is better than median; aborting...'
            break

        fit = population[selected]

        loc = np.mean(fit, axis=0)
        scale = np.std(fit, axis=0)

        t2 = dt.now()

        # report
        print 'generation %d: min: %.7f median: %.7f mean: %.7f max: %.7f time elapsed: %f' % (
            g, np.min(fitness), np.median(fitness), np.mean(fitness), np.max(fitness), (t2 - t1).total_seconds()
        )

        t1 = t2

    best_weights = population[np.argmax(fitness), :, :]

    return best_weights


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

    _ensemble, _population, predictions = load_population(clf, _population, X_train, y_train, X_val, y_val)

    _best_classifiers, _best_features, _best_predictions = select(_ensemble, _population, predictions, X_val, y_val)

    # _best_accs = np.sum(_best_hit_or_miss, axis=1) / float(_best_hit_or_miss.shape[1])

    _best_weights = integrate(
        _best_predictions, y_val,
        n_individuals=100, n_generations=100,
        models=_best_classifiers,
        features=_best_features,
        X_test=X_test,
        y_test=y_test
    )

    '''
        Now testing
    '''

    test_accuracy = __check_accuracy__(_best_weights, _best_classifiers, _best_features, X_test, y_test)
    print 'test accuracy: %.2f' % test_accuracy


if __name__ == '__main__':
    main()
