import numpy as np
import operator as op
from collections import Counter
import itertools as it
import pandas as pd


def check_distribution(ensemble, features, X, y):
    n_classifiers = len(ensemble)

    preds = np.empty((n_classifiers, y.shape[0]), dtype=np.float32)

    X_features = X.columns

    for i, classifier, features in it.izip(np.arange(n_classifiers), ensemble, features):
        preds[i] = classifier.predict(X[X_features[features]])

    print 'distribution of votes per instance in validation set:'

    counts = map(lambda x: Counter(preds[:, x]), xrange(y.shape[0]))

    df = pd.DataFrame(counts, dtype=np.float32)
    df.fillna(0., inplace=True)
    print df


def load_population(base_classifier, population, X_train, y_train, X_val, y_val):
    X_features = X_train.columns

    n_classifiers, n_attributes = population.shape

    ensemble = np.empty(n_classifiers, dtype=np.object)
    hit_or_miss = np.empty((n_classifiers, X_val.shape[0]), dtype=np.bool)

    for j in xrange(n_classifiers):  # number of base classifiers
        selected_features = X_features[population[j]]
        ensemble[j], hit_or_miss[j] = __get_classifier__(
            base_classifier, selected_features,
            X_train, y_train, X_val, y_val
        )
        if j % 50 == 0:
            print 'Loaded %d classifiers' % j

    return ensemble, population, hit_or_miss


def __get_classifier__(clf, selected_features, X_train, y_train, X_val, y_val):
    model = clf(random_state=0)

    try:
        model = model.fit(X_train[selected_features], y_train)
        preds = model.predict(X_val[selected_features])

        hit_or_miss = y_val == preds

    except ValueError:  # train set is empty
        hit_or_miss = np.zeros(y_val.shape[0])

    return model, hit_or_miss


def a_dominates_b(a, b):
    """

    :param a: list of performance in the n objective functions of individual a
    :param b: list of performance in the n objective functions of individual b
    :return: -1 if b dominates a, +1 if the opposite, and 0 if there is no dominance
    """
    a_dominates = any(a > b) and all(a >= b)
    b_dominates = any(b > a) and all(b >= a)

    res = (a_dominates * 1) + (b_dominates * -1)
    return res


def crowding_distance_assignment(_set):
    """
    Worst case scenario for this function: O(m * H * log(H)), where m
    is the number of objectives and H the size of population.

    :type _set: numpy.ndarray
    :param _set: An numpy.ndarray with m dimensions,
        where the first value in the tuple is the quality of the solution
        in the first objective, and so on and so forth.
    :return: A list of crowding distances.
    """

    n_individuals, n_objectives = _set.shape
    crowd_dists = np.zeros(n_individuals, dtype=np.float32)

    for objective in xrange(n_objectives):
        _set_obj = sorted(_set, key=lambda x: x[objective])
        crowd_dists[[0, -1]] = [np.inf, np.inf]
        for i in xrange(1, n_individuals - 1):
            crowd_dists[i] = crowd_dists[i] + (_set_obj[i + 1][objective] - _set_obj[i - 1][objective])

    return crowd_dists


def crowding_distance_sort(_set, indices):
    crowd_dists = crowding_distance_assignment(_set[indices])
    _sorted_indices = map(
        lambda x: x[1],
        sorted(zip(crowd_dists, indices), key=lambda x: x[0], reverse=False)
    )
    return _sorted_indices


def get_fronts(pop):
    n_individuals, n_objectives = pop.shape

    added = np.zeros(n_individuals, dtype=np.bool)
    dominated = np.zeros(n_individuals, dtype=np.int32)
    dominates = [[] for x in xrange(n_individuals)]
    fronts = []

    cur_front = 0

    for i in xrange(n_individuals):
        for j in xrange(i + 1, n_individuals):
            res = a_dominates_b(pop[i], pop[j])
            if res == 1:
                dominated[j] += 1
                dominates[i] += [j]
            elif res == -1:
                dominated[i] += 1
                dominates[j] += [i]

    while sum(added) < n_individuals:
        _where = (dominated == 0)

        if _where.max() == 0:  # no individuals that are not dominated by any other; add remaining
            fronts += [np.flatnonzero(added == 0)]  # TODO sort based on non-crowding distance!
            break

        added[_where] = 1

        current_front = np.flatnonzero(_where)

        # TODO sort based on crowding distance!

        fronts += [crowding_distance_sort(pop, current_front)]
        dominated[_where] = -1

        _chain = set(reduce(op.add, map(lambda x: dominates[x], _where)))

        for k in _chain:
            dominated[k] -= 1

        cur_front += 1

    # TODO must sort individuals based on their crowding distance!
    return fronts