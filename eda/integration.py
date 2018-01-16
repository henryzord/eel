import copy
from datetime import datetime as dt

import numpy as np
from sklearn.metrics import accuracy_score

from eda import Ensemble


def __get_elite__(P_fitness, A=None):
    median = np.median(P_fitness)

    if A is None:
        A = P_fitness < median
    else:
        A[:] = P_fitness < median

    return A


def __update__(P, A, loc):
    """

    :param A:
    :param P:
    :param loc: numpy.ndarray
    :return:
    """
    loc[:] = 0.

    n_elite = np.count_nonzero(A)

    for i in xrange(len(A)):
        if A[i]:
            loc += P[i].voting_weights

    loc[:] /= float(n_elite)

    return loc


def __get_best_individual__(P, P_fitness):
    return P[np.argmin(P_fitness)]  # type: Ensemble


def __save__(reporter, g, P, loc, scale):
    """

    :param reporter: eda.Reporter
    :param g:
    :param P:
    :param loc:
    :param scale:
    :return:
    """

    try:
        reporter.save_metrics(integrate, g, P)
        reporter.save_population(integrate, g, P)
        reporter.save_gm(integrate, g, loc, scale)
    except AttributeError:
        pass


def integrate(ensemble, n_individuals=100, n_generations=100, reporter=None):
    """
    Select an ensemble of classifiers.

    :param n_individuals: optional - number of individuals. Defaults to 100.
    :param n_generations: optional - number of generations. Defaults to 100.
    :type reporter: eda.Reporter
    :param reporter: optional - reporter for storing data about the evolutionary process.
        Defaults to None (no recording).
    :return: The best combination of base classifiers found.
    """
    n_classifiers = ensemble.n_classifiers
    n_classes = ensemble.n_classes

    scale = 0.25
    decay = scale / float(n_generations)
    loc = np.random.normal(loc=1., scale=scale, size=(n_classifiers, n_classes)).astype(dtype=np.float32)

    P = [copy.deepcopy(ensemble) for i in xrange(n_individuals)]
    P_fitness = np.empty(n_individuals, dtype=np.float32)
    A = np.zeros(n_individuals, dtype=np.int32)

    t1 = dt.now()

    last_median = 0
    streak = 0
    max_streak = 5

    __save__(reporter, 0, [ensemble], loc, scale)

    g = 1
    while g < n_generations:
        for i in xrange(n_individuals):
            if not A[i]:
                for j in xrange(P[i].n_classifiers):
                    for c in xrange(n_classes):
                        P[i].voting_weights[j] = np.clip(np.random.normal(loc=loc[j], scale=scale), a_min=0., a_max=1.)

                train_probs = P[i].predict_prob(P[i].X_train)
                argtrain = np.argmax(train_probs, axis=1)

                argwrong_train = np.flatnonzero(argtrain != P[i].y_train)
                wrong_train = np.max(train_probs[argwrong_train, :], axis=1)
                P_fitness[i] = np.sum(wrong_train)

        A = __get_elite__(P_fitness, A=A)
        best_individual = __get_best_individual__(P, P_fitness)  # type: Ensemble

        __save__(reporter, g, P, loc, scale)

        ensemble_train_acc = accuracy_score(ensemble.y_train, best_individual.predict(ensemble.X_train))
        dfd = best_individual.dfd(ensemble.X_train, ensemble.y_train)

        median = np.median(P_fitness, axis=0)

        if np.max(A) == 0:
            break
        if streak >= max_streak:
            break

        condition = (abs(last_median - median) < 0.01)
        streak = (streak * condition) + condition
        last_median = median

        loc = __update__(P, A, loc)
        scale -= decay

        t2 = dt.now()

        print 'generation %2.d: ens val acc: %.4f dfd: %.4f median: %.4f time elapsed: %f' % (
            g, ensemble_train_acc, dfd, median, (dt.now() - t1).total_seconds()
        )

        t1 = t2
        g += 1

    best_individual = __get_best_individual__(P, P_fitness)  # type: Ensemble

    __save__(reporter, g, P, loc, scale)

    return best_individual
