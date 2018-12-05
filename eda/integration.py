"""
Proper EDA script.
"""

import copy
import random
from datetime import datetime as dt

import numpy as np
from sklearn.metrics import accuracy_score

from eda import Ensemble


def __get_elite__(P_fitness, A=None):
    """
    From the set of individuals of the current population, selects the elite individuals based on whether
    their fitness surpasses the median or not.

    :param P_fitness: Fitness of population.
    :param A: Former elite population boolean array.
    :return: Updated elite population boolean array.
    """

    median = np.median(P_fitness)

    if A is None:
        A = P_fitness < median
    else:
        A[:] = P_fitness < median

    return A


def __update__(P, A, loc):
    """
    Update probabilistic graphical model based on elite population.

    :param A: A boolean array denoting whether the individual in that index is from the elite or not.
    :param P: The proper population of the EDA.
    :param loc: Former means of the probabilistic graphical model.
    :return: Updated mean for probabilistic graphical model.
    """
    loc[:] = 0.

    n_elite = np.count_nonzero(A)

    for i in range(len(A)):
        if A[i]:
            loc += P[i].voting_weights

    loc[:] /= float(n_elite)

    return loc


def __get_best_individual__(P, P_fitness):
    """
    Returns best individual from the population.

    :param P: Proper population of individuals.
    :param P_fitness: Fitness of population.
    :return: The best individual from the population.
    """
    return P[np.argmin(P_fitness)]  # type: Ensemble


def __save__(reporter, generation, A, P, P_fitness, loc, scale):
    """
    Saves metadata from EDA.

    :param reporter: eda.Reporter
    :param generation: generation index.
    :param P: Population of individuals.
    :param loc: Mean of probabilistic graphical model variables' PMF.
    :param scale: Std deviation of probabilistic graphical model variables' PMF.
    """

    try:
        reporter.save_population(generation=generation, elite=A, ensembles=P, P_fitness=P_fitness)
        reporter.save_gm(generation, loc, scale)
    except AttributeError:
        pass


def integrate(ensemble, n_individuals=100, n_generations=100, use_weights=True, reporter=None, verbose=True):
    """
    Optimize voting weights for an ensemble of base classifiers.

    :type ensemble: eda.Ensemble
    :param ensemble: ensemble of base classifiers.
    :type n_individuals: int
    :param n_individuals: optional - number of individuals. Defaults to 100.
    :type n_generations: int
    :param n_generations: optional - number of generations. Defaults to 100.
    :type use_weights: bool
    :param use_weights: Whether to use former weights from AdaBoost or not.
    :type reporter: reporter.EDAReporter
    :param reporter: optional - reporter for storing data about the evolutionary process.
        Defaults to None (no recording).
    :type verbose: bool
    :verbose: whether to output to console. Defaults to true.
    :return: The same ensemble, with optimized voting weights.
    """
    n_classifiers = ensemble.n_classifiers

    # overrides prior seed
    np.random.seed(None)
    random.seed(None)

    scale = 0.25
    decay = scale / float(n_generations)


    loc = np.empty((n_classifiers,1), dtype=np.float32)
    loc[:] = ensemble.voting_weights[:]


    P = []
    for i in range(n_individuals):
        P += [copy.deepcopy(ensemble)]
    P_fitness = np.empty(n_individuals, dtype=np.float32)
    A = np.zeros(n_individuals, dtype=np.int32)

    t1 = dt.now()

    last_median = 0
    streak = 0
    max_streak = 5

    ensemble_train_acc = accuracy_score(ensemble.y_train, ensemble.predict(ensemble.X_train))
    dfd = ensemble.dfd(ensemble.X_train, ensemble.y_train)

    print('generation %02.d: ens val acc: %.4f dfd: %.4f time elapsed: %f' % (
        -1, ensemble_train_acc, dfd, (dt.now() - t1).total_seconds()
    ))

    __save__(
        reporter=reporter, generation=-1, A=[0], P=[ensemble], P_fitness=[0], loc=loc, scale=scale
    )

    g = 0
    while g < n_generations:
        for i in range(n_individuals):
            if not A[i]:
                teste =  P[i].resample_voting_weights(loc=loc, scale=scale)
                P[i] = teste

                train_probs = P[i].predict_proba(P[i].X_train)
                argtrain = np.argmax(train_probs, axis=1)

                argwrong_train = np.flatnonzero(argtrain != P[i].y_train)
                wrong_train = np.max(train_probs[argwrong_train, :], axis=1)
                P_fitness[i] = np.sum(wrong_train)

        A = __get_elite__(P_fitness, A=A)
        best_individual = __get_best_individual__(P, P_fitness)  # type: Ensemble

        __save__(
            reporter=reporter, generation=g, A=A, P=P, P_fitness=P_fitness, loc=loc, scale=scale
        )

        ensemble_train_acc = accuracy_score(ensemble.y_train, best_individual.predict(ensemble.X_train))
        dfd = best_individual.dfd(ensemble.X_train, ensemble.y_train)

        median = float(np.median(P_fitness, axis=0))  # type: float

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

        if verbose:
            print('generation %02.d: ens val acc: %.4f dfd: %.4f median: %.4f time elapsed: %f' % (
                g, ensemble_train_acc, dfd, median, (t2 - t1).total_seconds()
            ))

        t1 = t2
        g += 1

    best_individual = __get_best_individual__(P, P_fitness)  # type: Ensemble

    __save__(
        reporter=reporter, generation=g, A=A, P=P, P_fitness=P_fitness, loc=loc, scale=scale
    )

    return best_individual
