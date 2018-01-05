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
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from utils import flatten


class Ensemble(object):
    def __init__(
            self,
            X_train, X_val, y_train, y_val,
            n_classifiers=None, classifiers=None, base_classifier=None,
            n_features=None, features=None,
            activated=None, voting_weights=None,
    ):
        """

        :type X_train: pandas.DataFrame
        :param X_train:
        :type X_val: pandas.DataFrame
        :param X_val:
        :param y_train:
        :param y_val:
        :param classifiers: A list of classifiers that are in compliance
            with the (fit, predict) interface of sklearn classifiers.
        :param features: An matrix where each row is a classifier and each
            column denotes the absence or presence of that attribute for the given classifier.
        :param activated: A boolean array denoting the activated classifiers.
        :param voting_weights: An matrix of shape (n_classifiers, n_classes).
        """

        assert isinstance(X_train, pd.DataFrame), TypeError('X_train must be a pandas.DataFrame!')
        assert isinstance(X_val, pd.DataFrame), TypeError('X_val must be a pandas.DataFrame!')

        scratch = ((n_classifiers is not None) and (base_classifier is not None))

        if scratch:
            if X_train is None or X_val is None or y_train is None or y_val is None:
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
                (n_classifiers, n_features), dtype=np.int32
            )
        else:
            raise ValueError('Either a list of activated features or the number of features must be provided!')
        # finally:
        self.n_features = len(self.truth_features[0])

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.feature_names = self.X_train.columns

        if voting_weights is not None:
            self.voting_weights = voting_weights
            self.n_classes = self.voting_weights.shape[1]
        else:
            self.n_classes = len(np.unique(
                np.hstack((y_train, y_val))
            ))
            self.voting_weights = np.ones(
                (self.n_classifiers, self.n_classes), dtype=np.float32
            )

        if activated is not None:
            self.activated = activated
        else:
            self.activated = np.ones(self.n_classifiers, dtype=np.int32)

        n_instances_val = self.X_val.shape[0]
        n_instances_train = self.X_train.shape[0]

        self.train_preds = np.empty((n_classifiers, n_instances_train), dtype=np.int32)
        self.val_preds = np.empty((n_classifiers, n_instances_val), dtype=np.int32)

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

    def get_predictions(self, X, preds=None):
        """
        Given a list of classifiers and the features each one of them uses, returns a matrix of predictions for dataset X.

        :param X: A dataset comprised of instances and attributes.
        :param preds: optional - matrix where each row is a classifier and each column an instance.
        :return: An matrix where each row is a classifier and each column an instance in X.
        """

        X_features = X.columns

        n_activated = np.count_nonzero(self.activated)
        index_activated = np.flatnonzero(self.activated)

        if preds is None:
            preds = np.empty((n_activated, X.shape[0]), dtype=np.int32)

        for raw, j in enumerate(index_activated):  # number of base classifiers
            selected_features = X_features[np.flatnonzero(self.truth_features[j])]
            preds[raw, :] = self.classifiers[j].predict(X[selected_features])

        return preds

    def predict(self, X, preds=None):
        """

        :param X: A dataset comprised of instances and attributes.
        :param preds: optional - matrix where each row is a classifier and each column an instance.
        :return: An array where each position contains the ensemble prediction for that instance.
        """
        preds = self.get_predictions(X, preds=preds)

        n_classifiers, n_classes = self.voting_weights.shape
        n_classifiers, n_instances = preds.shape

        local_votes = np.empty(n_classes, dtype=np.float32)
        global_votes = np.empty(n_instances, dtype=np.int32)

        for i in xrange(n_instances):
            local_votes[:] = 0.

            for j in xrange(n_classifiers):
                local_votes[preds[j, i]] += self.voting_weights[j, preds[j, i]]

            global_votes[i] = np.argmax(local_votes)

        return global_votes

    def dfd(self, X, y, preds=None):
        preds = self.get_predictions(X, preds)
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
    def __init__(self, Xs, ys, set_names, fold, n_run, output_path, date=None, n_jobs=4):
        self.Xs = Xs
        self.ys = ys
        self.set_sizes = map(len, self.ys)
        self.set_names = set_names
        if date is None:
            self.date = str(dt.now())
        else:
            if isinstance(date, datetime.datetime):
                date = str(date)
            self.date = date
        self.run = n_run
        self.output_path = output_path
        self.manager = Manager()
        self.gm_lock = Lock()
        self.report_lock = Lock()
        self.population_lock = Lock()
        self.processes = []
        self._fold = fold

    @property
    def fold(self):
        return self._fold

    @fold.setter
    def fold(self, value):
        self._fold = value

    def __get_hash__(self, func):
        return hash(func.__name__ + str(self.fold))

    def save_accuracy(self, func, gen, ensembles):
        """

        :dtype func: function
        :param func: function that is calling this method.
        :dtype gen: int
        :param gen: current generation.
        :dtype ensembles: list
        :param ensembles:
        """

        self.__save_accuracy__(func, gen, ensembles, self.report_lock)
        # p = Process(
        #     target=self.__save_accuracy__, args=(
        #         func, gen, ensembles, self.report_lock
        #     )
        # )
        # self.processes += [p]
        # p.start()

    def save_population(self, func, ensembles):
        self.__save_population__(self.output_path, self.date, func, ensembles, self.population_lock)
        # p = Process(
        #     target=self.__save_population__,
        #     args=(self.output_path, self.date, func, population, self.population_lock)
        # )
        # self.processes += [p]
        # p.start()

    def save_gm(self, func, gen, gm):
        self.__save_gm__(func, gen, gm, self.gm_lock)
        # p = Process(
        #     target=self.__save_gm__, args=(
        #         func, gen, gm, self.gm_lock
        #     )
        # )
        # self.processes += [p]
        # p.start()

    def __save_gm__(self, func, gen, gm, lock):
        """

        :param func:
        :param gen:
        :param gm:
        :type lock: multiprocessing.Lock
        :param lock:
        :return:
        """

        lock.acquire()

        output = os.path.join(self.output_path, self.date + '_gm_' + func.__name__ + '.csv')
        pth = Path(output)

        if not pth.exists():
            with open(output, 'w') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow(['method', 'fold', 'run', 'generation'] + ['var' + str(x) for x in xrange(gm.size)])

        with open(output, 'a') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow([func.__name__, self.fold, self.run, str(gen)] + list(gm.ravel()))

        lock.release()

    def __save_accuracy__(self, caller, gen, ensembles, lock):
        """

        :dtype caller: function
        :param caller: function that is calling this method.
        :dtype gen: int
        :param gen: current generation.
        :dtype ensembles: list
        :param ensembles: A list of the ensembles which will have their accuracy recorded.
        :type lock: multiprocessing.Lock
        :param lock:
        :return:
        """

        lock.acquire()

        n_sets = len(self.Xs)

        n_ensembles = len(ensembles)
        accs = np.empty(n_sets * len(ensembles), dtype=np.float32)

        counter = 0
        for ensemble in ensembles:
            for j, (X, y) in enumerate(it.izip(self.Xs, self.ys)):
                classes = ensemble.predict(X)
                acc = accuracy_score(y, classes)
                accs[counter] = acc
                counter += 1

        output = os.path.join(self.output_path, self.date + '_' + 'report_' + caller.__name__ + '.csv')
        # if not created
        pth = Path(output)
        if not pth.exists():
            with open(output, 'w') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow(
                    ['method', 'fold', 'run', 'generation', 'individual', 'set_name', 'set_size', 'accuracy'])

        # raise NotImplementedError('not implemented yet!')
        counter = 0
        with open(output, 'a') as f:
            writer = csv.writer(f, delimiter=',')

            for i in xrange(n_ensembles):
                for j in xrange(n_sets):
                    writer.writerow(
                        [caller.__name__, self.fold, self.run, str(gen), str(i), self.set_names[j], self.set_sizes[j],
                         str(accs[counter])])
                    counter += 1

        lock.release()

    def __save_population__(self, output_path, date, func, ensembles, lock):
        """

        :param output_path:
        :param date:
        :param func:
        :param ensembles:
        :type lock: multiprocessing.Lock
        :param lock:
        :return:
        """

        lock.acquire()

        if func.__name__ == 'generate':
            dense = np.array(map(lambda x: x.tolist(), ensembles))
        else:
            dense = ensembles

        pd.DataFrame(dense).to_csv(
            os.path.join(output_path, date + '_' + 'population' + '_' + func.__name__ + '.csv'),
            sep=',',
            index=False,
            header=False
        )

        lock.release()

    def join_all(self):
        """
        Join all running processes.
        """

        for p in self.processes:
            p.join()


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


# def check_distribution(ensemble, features, X, y):
#     n_classifiers = len(ensemble)
#
#     preds = np.empty((n_classifiers, y.shape[0]), dtype=np.float32)
#
#     X_features = X.columns
#
#     for i, classifier, features in it.izip(np.arange(n_classifiers), ensemble, features):
#         preds[i] = classifier.predict(X[X_features[features]])
#
#     print 'distribution of votes per instance in evaluation set:'
#
#     counts = map(lambda x: Counter(preds[:, x]), xrange(y.shape[0]))
#
#     df = pd.DataFrame(counts, dtype=np.float32)
#     df.fillna(0., inplace=True)
#     print df


# ---------------------------------------------------#
# ----------- # pareto-related methods # ----------- #
# ---------------------------------------------------#

def __deprecated_select_operator__(A_truth, P_fitness, u, e):
    """
    WARNING: deprecated.

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
        _flat_list = flatten(_fronts)
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
