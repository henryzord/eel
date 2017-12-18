"""
Check

> Using Bayesian Networks for Selecting Classifiers in GP Ensembles

for a measure on diversity.
"""

from datetime import datetime as dt

import numpy as np
from bitarray import bitarray
from eda import Ensemble, get_fronts
from sklearn.metrics import accuracy_score

from utils import flatten


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
        self.n_objectives = 2

    def __get_fitness__(
            self, ensemble,
            P_fitness=None, pairwise_double_fault_train=None, pairwise_double_fault_val=None
    ):
        """
        First objective is accuracy. Second objective is double-fault.
        see 'Genetic Algorithms with diversity measures to build classifier systems' for references

        :type ensemble:
        :param ensemble: List of classifiers.
        :type P_fitness: numpy.ndarray
        :param P_fitness: matrix to store fitness values.
        :param pairwise_double_fault_train:
        :param pairwise_double_fault_val:
        :rtype: numpy.ndarray
        :return: Returns a tuple where the first item is the fitness in the first objective, and so on and so forth.
        """
        n_classifiers = ensemble.n_classifiers
        n_instances_val = self.y_val.shape[0]
        n_instances_train = self.y_train.shape[0]

        if P_fitness is None:
            P_fitness = np.empty((n_classifiers, self.n_objectives), dtype=np.float32)

        if pairwise_double_fault_train is None:
            pairwise_double_fault_train = np.empty((n_classifiers, n_classifiers), dtype=np.float32)

        if pairwise_double_fault_val is None:
            pairwise_double_fault_val = np.empty((n_classifiers, n_classifiers), dtype=np.float32)

        for i in xrange(n_classifiers):
            for j in xrange(i, n_classifiers):
                val_index = np.sum(np.logical_xor(
                    ensemble.val_preds[i] == self.y_val,
                    ensemble.val_preds[j] == self.y_val
                )) / float(n_instances_val)

                train_index = np.sum(np.logical_xor(
                    ensemble.train_preds[i] == self.y_train,
                    ensemble.train_preds[j] == self.y_train
                )) / float(n_instances_train)

                pairwise_double_fault_val[i, j] = val_index
                pairwise_double_fault_val[j, i] = val_index

                pairwise_double_fault_train[i, j] = train_index
                pairwise_double_fault_train[j, i] = train_index

            P_fitness[i, 0] = np.median(pairwise_double_fault_train[i, :])
            P_fitness[i, 1] = np.median(pairwise_double_fault_val[i, :])

        return P_fitness

    def __get_elite__(self, P_fitness, A=None):
        fronts = get_fronts(P_fitness)
        flat = flatten(fronts)
        n_individuals = len(P_fitness)
        A_index = flat[:(n_individuals/2)]
        if A is None:
            A = np.zeros(n_individuals, dtype=np.int32)

        A[A_index] = True
        return A

    def __sample__(self, A, P, gm, ensemble):
        """

        :param A:
        :param P:
        :param gm:
        :type ensemble: eda.Ensemble
        :param ensemble:
        :return:
        """

        for j in xrange(ensemble.n_classifiers):
            if not A[j]:
                for k in xrange(ensemble.n_features):
                    P[j][k] = np.random.choice(a=[0, 1], p=[1. - gm[k], gm[k]])

            feature_index = list(P[j])
            ensemble = ensemble.set_classifier(index=j, base_classifier=self.base_classifier, feature_index=feature_index)

        return ensemble

    def __update__(self, A, ensemble, gm, selection_strength=0.5):
        """
        Encodes a new graphical model based on a population of individuals in Pareto Fronts. Uses selection operator from
            Laumanns, Marco and Ocenasek, Jiri. Bayesian Optimization Algorithms for Multi-objective Optimization. 2002.

        :param A: a boolean array the size of the population, where True means that
            this individual is in the elite, and False otherwise.
        :param P: Proper population.
        :param P_fitness: Quality of elite individuals
        :return: New Graphical Model.
        """
        assert isinstance(A, np.ndarray), TypeError('A must be a list!')
        assert isinstance(ensemble, Ensemble), TypeError('ensemble must be an instance of eda.Ensemble!')

        # warnings.warn('WARNING: plotting!')
        # from matplotlib import pyplot as plt
        # from matplotlib import cm
        # plt.figure()
        # plt.scatter(P_fitness[:, 0], P_fitness[:, 1])
        # plt.xlim(0.9, 1)
        # plt.ylim(0.9, 1)
        # plt.show()

        gm[:] = 0.
        for i in xrange(ensemble.n_classifiers):
            if A[i]:
                gm += ensemble.get_genotype(i)

        gm /= np.count_nonzero(A)

        return gm, A

    def generate(self, n_classifiers=100, n_generations=100, selection_strength=0.5, reporter=None):
        """

        :param n_classifiers:
        :param n_generations:
        :type reporter: eda.Reporter
        :param reporter:
        :return:
        """
        ensemble = Ensemble.create_base(
            X_train=self.X_train,
            X_val=self.X_val,
            y_train=self.y_train,
            y_val=self.y_val,
            base_classifier=self.base_classifier,
            n_classifiers=n_classifiers,
            n_features=self.n_features,
        )

        initial_prob = 0.5
        gm = np.full(shape=self.n_features, fill_value=initial_prob, dtype=np.float32)

        pairwise_double_fault_train = np.empty((n_classifiers, n_classifiers), dtype=np.float32)
        pairwise_double_fault_val = np.empty((n_classifiers, n_classifiers), dtype=np.float32)

        P = [bitarray(self.n_features) for i in xrange(n_classifiers)]  # population
        P_fitness = np.empty((n_classifiers, self.n_objectives), dtype=np.float32)
        A = np.zeros(n_classifiers, dtype=np.bool)  # elite

        for g in xrange(n_generations):
            t1 = dt.now()

            ensemble = self.__sample__(A=A, P=P, gm=gm, ensemble=ensemble)
            P_fitness = self.__get_fitness__(
                ensemble, P_fitness=P_fitness,
                pairwise_double_fault_train=pairwise_double_fault_train,
                pairwise_double_fault_val=pairwise_double_fault_val
            )

            A = self.__get_elite__(P_fitness, A=A)

            medians = np.median(P_fitness, axis=0)

            gm, A = self.__update__(A, ensemble, gm, selection_strength=selection_strength)
            ensemble_val_preds = ensemble.predict(self.X_val, ensemble.val_preds)
            ensemble_val_acc = accuracy_score(self.y_val, ensemble_val_preds)

            dfd = Ensemble.distinct_failure_diversity(ensemble.val_preds, self.y_val)

            # try:
            reporter.save_accuracy(self.generate, g, [ensemble])
            reporter.save_population(self.generate, ensemble.features)
            reporter.save_gm(self.generate, g, gm)
            # except AttributeError:
            #     pass

            print 'generation %2.d: ens val acc: %.4f dfd: %.4f median: (%.4f, %.4f) time elapsed: %f' % (
                g, ensemble_val_acc, dfd, medians[0], medians[1], (dt.now() - t1).total_seconds()
            )

        try:
            reporter.save_population(self.generate, ensemble.features)
        except AttributeError:
            pass

        return ensemble
        # A = self.__get_elite__(P_fitness, A=A)
        # A_index = np.flatnonzero(A)

        # ensemble.activated = A
        # return A
        # features = np.array(map(lambda x: x.tolist(), ensemble.features))
        # return np.array(ensemble.classifiers)[A_index], features[A_index], P_fitness[A_index]
