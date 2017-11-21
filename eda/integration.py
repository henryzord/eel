import json
import warnings
from datetime import datetime as dt

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier as clf
from eda.dataset import path_to_sets

from eda.core import load_population, get_classes, __get_classifier__


def __ensemble_predict_matrix__(voting_weights, predictions):
    n_classifiers, n_classes = voting_weights.shape
    n_classifiers, n_instances = predictions.shape

    global_votes = np.empty((n_instances, n_classes), dtype=np.float32)

    for i in xrange(n_instances):
        global_votes[i, :] = 0.

        for j in xrange(n_classifiers):
            global_votes[i, predictions[j, i]] += voting_weights[j, predictions[j, i]]

    return global_votes


# -------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------- #


def integrate(
    features, classifiers, val_predictions, y_val,
    n_individuals=100, n_generations=100,
    save_every=5, reporter=None
):
    """

    :param val_predictions:
    :param y_val:
    :param n_individuals:
    :param n_generations:
    :param save_every:
    :type reporter: eda.Reporter
    :param reporter:
    :return:
    """

    classes = np.unique(y_val)
    n_classes = len(classes)

    n_classifiers, n_val_instances = val_predictions.shape

    population = np.empty((n_individuals, n_classifiers, n_classes), dtype=np.float32)
    fitness = np.empty(n_individuals, dtype=np.float32)

    std = 1.

    loc = np.random.uniform(size=(n_classifiers, n_classes), low=1., high=10.)
    scale = np.full((n_classifiers, n_classes), std, dtype=np.float32)

    t1 = dt.now()

    for g in xrange(n_generations):
        for i in xrange(n_individuals):
            for j in xrange(n_classifiers):
                for c in xrange(n_classes):
                    population[i, j, c] = np.random.normal(loc=loc[j, c], scale=scale[j, c])

            population[i] = np.clip(population[i], a_min=0., a_max=1.)

            y_val_pred = get_classes(population[i], val_predictions)
            fitness[i] = accuracy_score(y_val, y_val_pred)

        try:
            reporter.save_accuracy(integrate, g, population, features, classifiers)
            reporter.save_population(integrate, population, g, save_every)
        except AttributeError:
            pass

        # update
        median = np.median(fitness)
        selected = fitness > median

        t2 = dt.now()

        # report
        print 'generation %d: min: %.7f median: %.7f mean: %.7f max: %.7f time elapsed: %f' % (
            g, np.min(fitness), np.median(fitness), np.mean(fitness), np.max(fitness), (t2 - t1).total_seconds()
        )

        t1 = t2

        if np.count_nonzero(selected) == 0:
            break

        fit = population[selected]
        loc = np.mean(fit, axis=0)

    try:
        reporter.save_population(integrate, population)
    except AttributeError:
        pass

    return population[np.argmax(fitness)]


def main():
    X_train, y_train, X_val, y_val, X_test, y_test = path_to_sets(
        '/home/henry/Projects/eel/datasets/ionosphere/ionosphere_full.arff',
        train_size=0.5, val_size=0.25, test_size=0.25, random_state=None
    )

    gen_file = pd.read_csv(
        '/home/henry/Projects/eel/metadata/2017-11-17 13:08:02.243386_population_generate.csv', dtype=np.bool
    )

    gen_pop = []
    features = []
    for i in xrange(len(gen_file)):
        features += [gen_file.iloc[i].values]
        gen_pop += [__get_classifier__(clf, np.flatnonzero(features[i]), X_train, y_train, X_val)]

    models, preds = zip(*gen_pop)
    features, models, preds = np.array(features), np.array(models), np.array(preds)

    sel_file = pd.read_csv(
        '/home/henry/Projects/eel/metadata/2017-11-17 13:08:02.243386_population_eda_select.csv', dtype=np.bool
    )

    int_pop = integrate(
        features, models, preds, y_val,
        n_individuals=100, n_generations=100,
        save_every=5, reporter=None
    )


if __name__ == '__main__':
    main()
