import copy
from datetime import datetime as dt

import numpy as np
from sklearn.metrics import accuracy_score

from eda import Ensemble


def __get_elite__(P_fitness, A=None):
    median = np.median(P_fitness)

    if A is None:
        A = P_fitness > median
    else:
        A[:] = P_fitness > median

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


def integrate(ensemble, X_val, y_val, n_individuals=100, n_generations=100, reporter=None):
    """
    Select an ensemble of classifiers.

    :param X_val: X for validation set.
    :param y_val: y for validation set.
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

    for g in xrange(n_generations):
        for i in xrange(n_individuals):
            if not A[i]:
                for j in xrange(P[i].n_classifiers):
                    for c in xrange(n_classes):
                        P[i].voting_weights[j][c] = np.clip(np.random.normal(loc=loc[j][c], scale=scale), a_min=0., a_max=1.)

                val_probs = P[i].predict_prob(X_val, preds=P[i].val_preds)
                train_probs = P[i].predict_prob(P[i].X_train, preds=P[i].train_preds)
                argval = np.argmax(val_probs, axis=1)
                argtrain = np.argmax(train_probs, axis=1)
                # argwrong, argright = np.flatnonzero(arg != y_val), np.flatnonzero(arg == y_val)

                argright_val = np.flatnonzero(argval == y_val)
                argright_train = np.flatnonzero(argtrain == P[i].y_train)
                right_val = np.max(val_probs[argright_val, :], axis=1)
                right_train = np.max(train_probs[argright_train, :], axis=1)
                # wrong = np.max(val_probs[argwrong, :], axis=1)
                P_fitness[i] = (np.sum(right_val) + np.sum(right_train)) / 2.

        A = __get_elite__(P_fitness, A=A)
        best_individual = P[np.argmax(P_fitness)]  # type: Ensemble

        # raise NotImplementedError('get best individual!')

        try:
            reporter.save_accuracy(integrate, g, P)
            reporter.save_population(integrate, best_individual.voting_weights)
            reporter.save_gm(integrate, g, loc)
        except AttributeError:
            pass

        ensemble_val_acc = accuracy_score(y_val, best_individual.predict(X_val, preds=best_individual.val_preds))
        dfd = best_individual.dfd(X_val, y_val, preds=best_individual.val_preds)

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
            g, ensemble_val_acc, dfd, median, (dt.now() - t1).total_seconds()
        )

        t1 = t2

    # A = __get_elite__(P_fitness, A=A)
    best_individual = P[np.argmax(P_fitness)]  # type: Ensemble

    try:
        reporter.save_population(integrate, best_individual.voting_weights)
    except AttributeError:
        pass

    return best_individual
