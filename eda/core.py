import warnings

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


class DummyClassifier(object):
    def __init__(self):
        pass

    def fit(self, X_train, y_train):
        pass

    def predict(self, X_test):
        return np.ones(X_test.shape[0], dtype=np.int32) * -1


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
    """
    :param clf:
    :param selected_features: Only the features selected for the given classifier
    :param X_train:
    :param y_train:
    :param X_val:
    :rtype: tuple
    :return: model, preds
    """

    assert isinstance(X_train, pd.DataFrame), TypeError('X_train must be a pandas.DataFrame!')
    assert isinstance(X_val, pd.DataFrame), TypeError('X_val must be a pandas.DataFrame!')

    model = clf(random_state=0)

    classes = np.unique(y_train)

    if len(selected_features) <= 0:
        model = DummyClassifier()
    else:
        model = model.fit(X_train[selected_features], y_train)
    preds = model.predict(X_val[selected_features])

    return model, preds

# ---------------------------------------------------#
# ----------- # pareto-related methods # ----------- #
# ---------------------------------------------------#


def get_flat_list(array):
    """
    Returns a flat list, made out of a list of lists.

    :param array: A list of lists.
    :return: A flat list.
    """

    return [item for sublist in array for item in sublist]


def __pareto_encode_gm__(A, P, P_fitness, select_strength=0.5):
    """
    Encodes a new graphical model based on a population of individuals in Pareto Fronts. Uses selection operator from
        Laumanns, Marco and Ocenasek, Jiri. Bayesian Optimization Algorithms for Multi-objective Optimization. 2002.

    :param A: a boolean array the size of the population, where True means that
        this individual is in the elite, and False otherwise.
    :param P: Proper population.
    :param P_fitness: Quality of elite individuals
    :return: New Graphical Model.
    """
    assert isinstance(A, np.ndarray), TypeError('A must be a list!')
    assert isinstance(P, list), TypeError('P must be a list!')

    # either this
    fronts = get_fronts(P_fitness)
    _flat = get_flat_list(fronts)
    A_ = _flat[:int(len(P_fitness) * select_strength)]

    # or this
    # A_ = select_operator(
    #     A_truth=A,
    #     P_fitness=P_fitness,
    #     u=int(0.25 * len(P_fitness)),
    #     e=0.1
    # )

    A[:] = False
    A[A_] = True

    gm = np.sum([P[i].tolist() for i in A_], axis=0) / float(len(A_))
    return gm, A


def select_operator(A_truth, P_fitness, u, e):
    """

    :type A_truth: np.ndarray
    :param A_truth: A boolean array denoting the elite individuals.
    :type P_fitness: numpy.ndarray
    :param P_fitness: An array denoting the quality of solutions. The first A elements are the former elite,
        and the following P elements the current population.
    :type u: int
    :param u: minimum size of new elite populaiton
    :type e: float
    :param e: approximation factor: the smaller it is, the small the tolerance for difference is. Resides within [0, 1].
    :return: Selected individuals.
    """
    log2e = np.log2(e)

    P = set(np.flatnonzero(A_truth == False))
    A = set(np.flatnonzero(A_truth == True))

    floor = np.floor(np.log2(P_fitness) / log2e)

    for x in P:
        similar = np.multiply.reduce(
            floor[x] == floor[list(A)],
            axis=1
        )
        B = set(np.flatnonzero(similar))  # B_index is the set of solutions similar to x_index

        if len(B) == 0:  # if there is no solution remotely equal to x
            A |= {x}  # add x to the new elite
        elif any([(a_dominates_b(P_fitness[y], P_fitness[x]) < 1) for y in B]):
            # if there is a solution in new_A_index that dominates x
            # A = A - (B | {x})
            A = A - (B | {x})

    A_ = set()
    for y in A:
        add = True
        for z in A:
            if a_dominates_b(P_fitness[y], P_fitness[z]) < 1:
                add = False
                break
        if add:
            A_ |= {y}

    D = A - A_

    if len(A_) < u:
        _fronts = get_fronts(P_fitness[list(D | A_)])
        _flat_list = get_flat_list(_fronts)
        for ind in _flat_list:
            A_ |= {ind}
            if len(A_) >= u:
                break

    return list(A_)


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
    assert type(a) in [np.void, np.ndarray, list], TypeError('invalid type for a and b! Must be either lists or void objects!')

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

    if (distinct_failures_count > 0) and (n_classifiers > 1):
        for j in xrange(1, n_classifiers + 1):
            dfd += (float(n_classifiers - j)/float(n_classifiers - 1)) * \
                   (float(distinct_failures[j]) / distinct_failures_count)

    return dfd
