import numpy as np
from datetime import datetime as dt

from sklearn.metrics import accuracy_score
from eda import Ensemble
import copy


def __get_elite__(P_fitness, A=None):
    median = np.median(P_fitness, axis=0)

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

    loc = np.random.normal(loc=0.5, scale=1., size=(n_classifiers, n_classes)).astype(dtype=np.float32)
    scale = 0.5

    P = [copy.deepcopy(ensemble) for i in xrange(n_individuals)]
    P_fitness = np.empty(n_individuals, dtype=np.float32)
    A = np.zeros(n_individuals, dtype=np.int32)

    t1 = dt.now()

    for g in xrange(n_generations):
        for i in xrange(n_individuals):
            if not A[i]:
                for j in xrange(P[i].n_classifiers):
                    for c in xrange(n_classes):
                        P[i].voting_weights[j][c] = np.random.normal(loc=loc[j][c], scale=scale)

                P_fitness[i] = P[i].dfd(X=X_val, y=y_val, preds=P[i].val_preds)

        try:
            reporter.save_accuracy(integrate, g, P)
            reporter.save_population(integrate, P[np.argmax(P_fitness)].voting_weights)
            reporter.save_gm(integrate, g, loc)
        except AttributeError:
            pass

        A = __get_elite__(P_fitness, A=A)

        best_individual = P[np.argmax(P_fitness)]

        ensemble_val_acc = accuracy_score(y_val, best_individual.predict(X_val, preds=best_individual.val_preds))
        dfd = best_individual.dfd(X_val, y_val, preds=best_individual.val_preds)

        if np.max(A) == 0:
            break

        medians = np.median(P_fitness, axis=0)
        loc = __update__(P, A, loc)

        t2 = dt.now()

        print 'generation %2.d: ens val acc: %.4f dfd: %.4f median: %.4f time elapsed: %f' % (
            g, ensemble_val_acc, dfd, medians, (dt.now() - t1).total_seconds()
        )

        t1 = t2

    best_individual = P[np.argmax(P_fitness)]

    try:
        reporter.save_population(integrate, P[np.argmax(P_fitness)].voting_weights)
    except AttributeError:
        pass

    return best_individual
