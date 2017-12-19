import numpy as np
from datetime import datetime as dt

from sklearn.metrics import accuracy_score

from eda import Ensemble
import copy
from eda.selection.graphical_model import GraphicalModel


def get_selection_fitness(P, y_true):
    pass


def __get_elite__(P_fitness, A=None):
    median = np.median(P_fitness, axis=0)

    if A is None:
        A = P_fitness > median
    else:
        A[:] = P_fitness > median
    return A


def __update__(P, A, gm):
    """

    :param A:
    :param P:
    :param gm: eda.selection.graphical_model.GraphicalModel
    :return:
    """

    gm = gm.update(P, A)
    return gm


def select(ensemble, X_val, y_val, n_individuals=100, n_generations=100, reporter=None):
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

    gm = GraphicalModel(variable_names=range(n_classifiers), available_values=[0, 1])
    P = [copy.deepcopy(ensemble) for i in xrange(n_individuals)]
    P_fitness = np.empty(n_individuals, dtype=np.float32)
    A = np.zeros(n_individuals, dtype=np.int32)

    t1 = dt.now()

    for g in xrange(n_generations):
        for i in xrange(n_individuals):
            if not A[i]:
                P[i] = gm.sample(P[i])
                P_fitness[i] = P[i].dfd(X=X_val, y=y_val, preds=P[i].val_preds)

        # try:
        reporter.save_accuracy(select, g, P)
        reporter.save_population(select, P[np.argmax(P_fitness)].features)
        # reporter.save_gm(select, g, gm)  # TODO save GM in later iterations!
        # except AttributeError:
        #     pass

        A = __get_elite__(P_fitness, A=A)

        best_individual = P[np.argmax(P_fitness)]
        ensemble_val_acc = accuracy_score(y_val, best_individual.predict(X_val, preds=best_individual.val_preds))
        dfd = best_individual.dfd(X_val, y_val, preds=best_individual.val_preds)

        if np.max(A) == 0:
            break

        medians = np.median(P_fitness, axis=0)
        gm = __update__(P, A, gm)

        t2 = dt.now()

        print 'generation %2.d: ens val acc: %.4f dfd: %.4f median: %.4f time elapsed: %f' % (
            g, ensemble_val_acc, dfd, medians, (dt.now() - t1).total_seconds()
        )

        t1 = t2

    best_individual = P[np.argmax(P_fitness)]

    try:
        reporter.save_population(select, best_individual.features)
    except AttributeError:
        pass

    return best_individual
