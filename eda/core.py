import numpy as np
import operator as op
from collections import Counter
import itertools as it
import pandas as pd


class DummyIterator(object):
    def __init__(self, value, length, reset=True):
        self.current = 0
        self.length = length
        self.value = value
        self.reset = reset

    def __len__(self):
        return self.length

    def __iter__(self):
        return self

    def next(self):
        if self.current < self.length:
            self.current += 1
            return self.value
        else:
            if self.reset:
                self.current = 0
            raise StopIteration


def get_predictions(classifiers, features, X):
    """
    Given a list of classifiers and the features each one of them uses, returns a matrix of predictions for dataset X.

    :param classifiers: A list of classifiers that are in compliance
        with the (fit, predict) interface of sklearn classifiers.
    :param features: An matrix where each row is a classifier and each
        column denotes the absence or presence of that attribute for the given classifier.
    :param X: A dataset comprised of instances and attributes.
    :return: An matrix where each row is a classifier and each column an instance in X.
    """

    n_classifiers = len(features)

    X_features = X.columns

    preds = np.empty((n_classifiers, X.shape[0]), dtype=np.int32)

    for j in xrange(n_classifiers):  # number of base classifiers
        selected_features = X_features[features[j]]
        preds[j, :] = classifiers[j].predict(X[selected_features])

    return preds


def get_classes(voting_weights, preds):
    """

    :param voting_weights: An matrix of shape (n_classifiers, n_classes).
    :param preds: An matrix of shape (n_classifiers, n_instances).
    :return:
    """

    n_classifiers, n_classes = voting_weights.shape
    n_classifiers, n_instances = preds.shape

    local_votes = np.empty(n_classes, dtype=np.float32)
    global_votes = np.empty(n_instances, dtype=np.int32)

    for i in xrange(n_instances):
        local_votes[:] = 0.

        for j in xrange(n_classifiers):
            local_votes[preds[j, i]] += voting_weights[j, preds[j, i]]

        global_votes[i] = np.argmax(local_votes)

    return global_votes


def check_distribution(ensemble, features, X, y):
    n_classifiers = len(ensemble)

    preds = np.empty((n_classifiers, y.shape[0]), dtype=np.float32)

    X_features = X.columns

    for i, classifier, features in it.izip(np.arange(n_classifiers), ensemble, features):
        preds[i] = classifier.predict(X[X_features[features]])

    print 'distribution of votes per instance in evaluation set:'

    counts = map(lambda x: Counter(preds[:, x]), xrange(y.shape[0]))

    df = pd.DataFrame(counts, dtype=np.float32)
    df.fillna(0., inplace=True)
    print df


def load_population(base_classifier, population, X_train, y_train, X_val, y_val, verbose=True):
    X_features = X_train.columns

    n_classifiers, n_attributes = population.shape

    ensemble = np.empty(n_classifiers, dtype=np.object)
    predictions = np.empty((n_classifiers, X_val.shape[0]), dtype=np.int32)

    for j in xrange(n_classifiers):  # number of base classifiers
        selected_features = X_features[population[j]]
        ensemble[j], predictions[j] = __get_classifier__(
            base_classifier, selected_features,
            X_train, y_train, X_val
        )
        if j % 50 == 0 and verbose:
            print 'Loaded %d classifiers' % j

    return ensemble, population, predictions


def __get_classifier__(clf, selected_features, X_train, y_train, X_val):
    model = clf(random_state=0)

    classes = np.unique(y_train)

    try:
        model = model.fit(X_train[selected_features], y_train)
        preds = model.predict(X_val[selected_features])

    except ValueError:  # train set is empty
        preds = np.random.choice(classes, replace=True, size=X_train.shape[0])

    return model, preds

# ---------------------------------------------------#
# ----------- # pareto-related methods # ----------- #
# ---------------------------------------------------#


def __pareto_encode_gm__(population, fitness):
    """
    Encodes a new graphical model based on a population of individuals in Pareto Fronts.

    :param population:
    :param fitness:
    :return:
    """
    to_pick = []

    # where first position is the first front, second position is the first front, and so on
    fronts = get_fronts(fitness)
    # compress list of lists in a single list of ordered individuals, based on their non-dominated rank
    flat_list = [item for sublist in fronts for item in sublist]

    # start picking individuals from these fronts
    for i, ind in enumerate(flat_list):
        # if any(np.array(fitness[ind]) < np.array(medians)):
        #     break
        if i < (len(population) / 2):
            to_pick += [population[ind].tolist()]

    gm = np.sum(to_pick, axis=0) / float(len(to_pick))
    return gm


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
    """
    Based on a population of individuals, order them based on their Pareto dominance.

    :param pop: A matrix where rows are individuals and columns their fitness in each one of the objectives.
    :type pop: numpy.ndarray
    :return:
    """

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
            fronts += [np.flatnonzero(added == 0)]
            break

        added[_where] = 1

        current_front = np.flatnonzero(_where)

        fronts += [crowding_distance_sort(pop, current_front)]
        dominated[_where] = -1

        _chain = set(reduce(op.add, map(lambda x: dominates[x], _where)))

        for k in _chain:
            dominated[k] -= 1

        cur_front += 1

    return fronts
