import json

import numpy as np
import pandas as pd
from bitarray import bitarray
from sklearn.tree import DecisionTreeClassifier as clf

from eda.core import get_fronts, __get_classifier__
from eda.dataset import load_sets
from datetime import datetime as dt
from multiprocessing import Process
from core import check_distribution

'''
Check

Using Bayesian Networks for Selecting Classifiers in GP Ensembles

for a measure on diversity.
 
'''


def __save_population__(population):
    print '\tsaving population...'
    dense = np.array(map(lambda x: x.tolist(), population))

    pd.DataFrame(dense).to_csv('generation_population.csv', sep=',', index=False)


def get_fitness(ensemble, fitness, hit_or_miss, n_instances_val):
    """

    First objective is accuracy. Second objective is double-fault.
    see 'Genetic Algorithms with diversity measures to build classifier systems' for references

    :param ensemble: List of classifiers.
    :param fitness: matrix to store fitness values.
    :return: Returns a tuple where the first item is the fitness in the first objective, and so on and so forth.
    """
    n_classifiers = len(ensemble)

    pairwise_double_fault = np.empty((n_classifiers, n_classifiers), dtype=np.float32)

    for i in xrange(n_classifiers):
        fitness[i, 0] = np.sum(hit_or_miss[i, :]) / float(n_instances_val)

        for j in xrange(i, n_classifiers):
            hits = np.sum(np.logical_xor(hit_or_miss[i], hit_or_miss[j]))

            pairwise_double_fault[i, j] = hits / float(n_instances_val)
            pairwise_double_fault[j, i] = hits / float(n_instances_val)

        fitness[i, 1] = np.mean(pairwise_double_fault[i, :])

    return fitness


def __replace_population__(population, fitness, p, medians):
    fronts = get_fronts(fitness)
    # start picking individuals from these fronts

    to_pick = []

    flat_list = [item for sublist in fronts for item in sublist]
    for i, ind in enumerate(flat_list):
        # if any(np.array(fitness[ind]) < np.array(medians)):
        #     break
        if i < (len(population) / 2):
            to_pick += [population[ind].tolist()]

    p = np.mean(to_pick, axis=0).ravel()
    return p


def generate(X_train, y_train, X_val, y_val, base_classifier, n_classifiers=100, n_generations=100, save_at=10):
    X_features = X_train.columns
    n_features = len(X_features)
    n_objectives = 2  # accuracy and diversity

    n_instances_val = X_val.shape[0]

    initial_prob = 0.5
    p = np.full(shape=n_features, fill_value=initial_prob, dtype=np.float32)

    ensemble = np.empty(n_classifiers, dtype=np.object)

    # first column for accuracy, second for scalar double fault
    fitness = np.empty((n_classifiers, n_objectives), dtype=np.float32)
    hit_or_miss = np.empty((n_classifiers, n_instances_val), dtype=np.bool)

    # population
    population = [bitarray(n_features) for i in xrange(n_classifiers)]

    t1 = dt.now()

    for g in xrange(n_generations):
        for j in xrange(n_classifiers):
            for k in xrange(n_features):
                population[j][k] = np.random.choice(a=[0, 1], p=[1. - p[k], p[k]])

            selected_features = X_features[list(population[j])]
            ensemble[j], hit_or_miss[j] = __get_classifier__(
                base_classifier, selected_features, X_train, y_train, X_val, y_val
            )

        fitness = get_fitness(ensemble, fitness, hit_or_miss, n_instances_val)
        medians = np.median(fitness, axis=0)
        means = np.mean(fitness, axis=0)

        p = __replace_population__(population, fitness, p, medians)

        t2 = dt.now()

        if g % save_at == 0:
            Process(
                target=__save_population__,
                kwargs=dict(population=population)
            ).start()

        print 'generation %d: median: (%.4f, %.4f) mean: (%.4f, %.4f) time elapsed: %f' % (
            g, medians[0], medians[1], means[0], means[1], (t2 - t1).total_seconds()
        )
        t1 = t2

    __save_population__(population=population)

    dense = np.array(map(lambda x: x.tolist(), population))
    return ensemble, dense, fitness


def main():
    params = json.load(open('../params.json', 'r'))

    print 'loading datasets...'

    X_train, X_val, X_test, y_train, y_val, y_test = load_sets(
        params['train_path'],
        params['val_path'],
        params['test_path']
    )

    ensemble, population, accs = generate(
        X_train, y_train, X_val, y_val,
        base_classifier=clf,
        n_classifiers=200,
        n_generations=10
    )

    check_distribution(ensemble, population, X_test, y_test)

    print 'base learners x features:', population.shape
    print 'accuracies:'
    print 'min:\t %.7f' % np.min(accs)
    print 'median:\t %.7f' % np.median(accs)
    print 'mean:\t %.7f' % np.mean(accs)
    print 'max:\t %.7f' % np.max(accs)


if __name__ == '__main__':
    main()
