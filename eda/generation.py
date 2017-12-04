import warnings
from datetime import datetime as dt

import numpy as np
from bitarray import bitarray
from sklearn.metrics import accuracy_score

from core import __pareto_encode_gm__, get_classes, distinct_failure_diversity
from eda.core import __get_classifier__, DummyIterator

'''
Check

> Using Bayesian Networks for Selecting Classifiers in GP Ensembles

for a measure on diversity.
 
'''


class ConversorIterator(object):
    def __init__(self, population):
        self.current = 0
        self.length = len(population)
        self.population = population

    def __getitem__(self, item):
        return np.array(list(self.population[item]))

    def __iter__(self):
        return self

    def __len__(self):
        return self.length

    def next(self):
        if self.current < self.length:
            ind = np.array(list(self.population[self.current]))
            self.current += 1
            return ind
        else:
            raise StopIteration


class EnsembleGenerator(object):
    def __init__(self, X_train, y_train, X_val, y_val, base_classifier):
        self.y_val = y_val
        self.base_classifier = base_classifier
        self.X_val = X_val
        self.y_train = y_train
        self.X_train = X_train
        self.X_features = self.X_train.columns
        self.n_features = len(self.X_features)

    def get_generation_fitness(self, ensemble, fitness, val_predictions):
        """

        First objective is accuracy. Second objective is double-fault.
        see 'Genetic Algorithms with diversity measures to build classifier systems' for references

        :type ensemble:
        :param ensemble: List of classifiers.
        :type fitness: numpy.ndarray
        :param fitness: matrix to store fitness values.
        :type val_predictions: numpy.ndarray
        :param val_predictions: matrix where each row is a classifier and each column a prediction for that instance.
        :type y_val: numpy.ndarray
        :param y_val: array with real class for validation set.
        :rtype: numpy.ndarray
        :return: Returns a tuple where the first item is the fitness in the first objective, and so on and so forth.
        """
        n_classifiers = len(ensemble)

        n_instances_val = self.y_val.shape[0]

        pairwise_double_fault = np.empty((n_classifiers, n_classifiers), dtype=np.float32)

        for i in xrange(n_classifiers):
            fitness[i, 0] = accuracy_score(self.y_val, val_predictions[i, :])

            for j in xrange(i, n_classifiers):
                index = np.sum(np.logical_or(
                    val_predictions[i] == self.y_val,
                    val_predictions[j] == self.y_val
                )) / float(n_instances_val)

                pairwise_double_fault[i, j] = index
                pairwise_double_fault[j, i] = index

            # warnings.warn('WARNING: using min instead of mean!')
            fitness[i, 1] = np.mean(pairwise_double_fault[i, :])

        # fitness = normalize(fitness, axis=0, norm='max')  # normalize fitness
        return fitness

    def __sample__(self, A, P, P_fitness, gm, classifiers, val_preds):
        n_classifiers = len(classifiers)
        n_features = len(self.X_features)

        for j in xrange(n_classifiers):
            if not A[j]:
                for k in xrange(n_features):
                    P[j][k] = np.random.choice(a=[0, 1], p=[1. - gm[k], gm[k]])

            selected_features = self.X_features[list(P[j])]
            classifiers[j], val_preds[j] = __get_classifier__(
                self.base_classifier, selected_features, self.X_train, self.y_train, self.X_val
            )

        P_fitness = self.get_generation_fitness(classifiers, P_fitness, val_preds)

        return P, P_fitness, classifiers, val_preds

    def generate(self, n_classifiers=100, n_generations=100, selection_strength=0.5, save_every=5, reporter=None):
        """

        :param n_classifiers:
        :param n_generations:
        :param save_every:
        :type reporter: eda.Reporter
        :param reporter:
        :return:
        """
        n_objectives = 2  # accuracy and diversity
        # -- dummy weights -- #
        n_classes = len(np.unique(self.y_val))
        dummy_weights = np.ones((n_classifiers, n_classes), dtype=np.float32)
        dummy_weight_vector = DummyIterator(  # whole population of classifiers equals to one ensemble
            dummy_weights, length=1, reset=True
        )
        # -- dummy weights -- #

        n_instances_val = self.X_val.shape[0]

        initial_prob = 0.5
        gm_0 = np.full(shape=self.n_features, fill_value=initial_prob, dtype=np.float32)  # pareto multi-objective

        classifiers_0 = np.empty(n_classifiers, dtype=np.object)

        # first column for accuracy, second for scalar double fault
        P_fitness_0 = np.empty((n_classifiers, n_objectives), dtype=np.float32)
        val_preds_0 = np.empty((n_classifiers, n_instances_val), dtype=np.int32)

        # population
        P_0 = [bitarray(self.n_features) for i in xrange(n_classifiers)]

        A_0 = np.zeros(n_classifiers, dtype=np.bool)

        t1 = dt.now()

        g = 0
        while g < n_generations:
            t2 = dt.now()

            P_0, P_fitness_0, classifiers_0, val_preds_0 = self.__sample__(
                A_0, P_0, P_fitness_0, gm_0, classifiers_0, val_preds_0
            )

            ensemble_preds = get_classes(dummy_weights, val_preds_0)
            dfd = distinct_failure_diversity(val_preds_0, self.y_val)
            ensemble_acc = accuracy_score(self.y_val, ensemble_preds)

            medians = np.median(P_fitness_0, axis=0)

            gm_0, A_0 = __pareto_encode_gm__(A_0, P_0, P_fitness_0, select_strength=selection_strength)

            try:
                reporter.save_accuracy(self.generate, g, dummy_weight_vector, ConversorIterator(P_0), classifiers_0)
                reporter.save_population(self.generate, P_0, g, save_every)
                reporter.save_gm(self.generate, g, gm_0)
            except AttributeError:
                pass

            print 'generation %2.d: ens val acc: %.2f dfd: %.4f median: (%.4f, %.4f) time elapsed: %f' % (
                g, ensemble_acc, dfd, medians[0], medians[1], (t2 - t1).total_seconds()
            )
            t1 = t2
            g += 1

        try:
            reporter.save_population(self.generate, P_0)
        except AttributeError:
            pass

        features = np.array(map(lambda x: x.tolist(), P_0))
        return classifiers_0, features, P_fitness_0
