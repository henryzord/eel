from datetime import datetime as dt

import numpy as np

from eda import get_fronts


def select(
        features, classifiers, val_predictions, y_val,
        n_individuals=100, n_generations=100, reporter=None
):
    """
    Select an ensemble of classifiers.

    :param features:
    :param classifiers:
    :param val_predictions:
    :param y_val:
    :param n_individuals:
    :param n_generations:
    :type reporter: eda.Reporter
    :param reporter:
    :return:
    """

    n_classifiers, n_features = features.shape
    n_objectives = 2

    gm = np.full(n_classifiers, 0.5, dtype=np.float32)
    sel_pop = np.empty((n_individuals, n_classifiers), dtype=np.bool)
    fitness = np.empty((n_individuals, n_objectives), dtype=np.float32)

    n_classes = len(np.unique(y_val))

    dummy_weights = np.empty(
        (n_individuals, n_classifiers, n_classes),
        dtype=np.float32
    )

    t1 = dt.now()

    for g in xrange(n_generations):
        for i in xrange(n_individuals):
            for j in xrange(n_classifiers):
                sel_pop[i, j] = np.random.choice(a=[True, False], p=[gm[j], 1. - gm[j]])
                dummy_weights[i, j, :] = sel_pop[i, j]

            fitness[i, :] = get_selection_fitness(val_predictions[sel_pop[i]], y_val)

        try:
            reporter.save_accuracy(select, g, dummy_weights, features, classifiers)
            reporter.save_population(select, sel_pop, g)
        except AttributeError:
            pass

        medians = np.median(fitness, axis=0)
        means = np.mean(fitness, axis=0)

        # individuals with fitness equal or higher than median
        selected = np.multiply.reduce(fitness >= medians, axis=1)

        if np.count_nonzero(selected) == 0:
            print 'no individual is better than median; aborting...'
            break

        t2 = dt.now()

        print 'generation %2.d: median: (%.4f, %.4f) mean: (%.4f, %.4f) time elapsed: %f' % (
            g, medians[0], medians[1], means[0], means[1], (t2 - t1).total_seconds()
        )
        t1 = t2

        gm = __pareto_encode_gm__(selected, sel_pop, fitness)

    # medians = np.median(fitness, axis=0)
    # selected = np.multiply.reduce(fitness >= medians, axis=1)

    fronts = get_fronts(fitness)
    try:
        reporter.save_population(select, sel_pop)
    except AttributeError:
        pass

    # first individual from first front, which is the most sparse in comparison to other individuals
    raise NotImplementedError('not implemented yet!')
    return fronts[0][0]
