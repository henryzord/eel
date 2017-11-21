import csv
import itertools as it
import json
import time
from multiprocessing import Process, Manager, Lock

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier as clf

from core import get_predictions, get_classes
from eda.dataset import path_to_sets
from generation import generate
from integration import integrate
from selection import eda_select
import os
from datetime import datetime as dt
import pandas as pd
from pathlib2 import Path
from generation import generate
from selection import eda_select
from integration import integrate
import datetime


class Reporter(object):
    def __init__(self, Xs, ys, set_names, fold, n_run, output_path, date=None, n_jobs=4):
        self.Xs = Xs
        self.ys = ys
        self.set_sizes = map(len, self.ys)
        self.set_names = set_names
        if date is None:
            self.date = str(dt.now())
        else:
            if isinstance(date, datetime.datetime):
                date = str(date)
            self.date = date
        self.run = n_run
        self.output_path = output_path
        self.manager = Manager()
        self.gm_lock = Lock()
        self.report_lock = Lock()
        self.population_lock = Lock()
        self.processes = []
        self._fold = fold

    @property
    def fold(self):
        return self._fold

    @fold.setter
    def fold(self, value):
        self._fold = value

    def __get_hash__(self, func):
        return hash(func.__name__ + str(self.fold))

    def save_accuracy(self, func, gen, weights, features, classifiers):
        # self.__report__(func, gen, weights, features, classifiers, dict())
        p = Process(
            target=self.__save_accuracy__, args=(
                func, gen, weights, features, classifiers, self.report_lock
            )
        )
        self.processes += [p]
        p.start()

    def save_population(self, func, population, gen=1, save_every=1):
        if (gen > 0) and (gen % save_every == 0):
            # self.__save__(self.output_path, self.date, func, population, dict())
            p = Process(
                target=self.__save_population__,
                args=(self.output_path, self.date, func, population, self.population_lock)
            )
            self.processes += [p]
            p.start()

    def save_gm(self, func, gen, gm):
        p = Process(
            target=self.__save_gm__, args=(
                func, gen, gm, self.gm_lock
            )
        )
        self.processes += [p]
        p.start()

    def __save_gm__(self, func, gen, gm, lock):
        """

        :param func:
        :param gen:
        :param gm:
        :type lock: multiprocessing.Lock
        :param lock:
        :return:
        """

        lock.acquire()

        output = os.path.join(self.output_path, self.date + '_gm_' + func.__name__ + '.csv')
        pth = Path(output)

        if not pth.exists():
            with open(output, 'w') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow(['method', 'fold', 'run', 'generation'] + ['var' + str(x) for x in xrange(gm.size)])

        with open(output, 'a') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow([func.__name__, self.fold, self.run, str(gen)] + list(gm.ravel()))

        lock.release()

    def __save_accuracy__(self, func, gen, weights, features, classifiers, lock):
        """

        :param func:
        :param gen:
        :param weights:
        :param features:
        :param classifiers:
        :type lock: multiprocessing.Lock
        :param lock:
        :return:
        """

        lock.acquire()

        n_sets = len(self.Xs)
        n_individuals = len(weights)

        accs = np.empty(n_sets * n_individuals, dtype=np.float32)

        counter = 0
        for weight_set in weights:
            for j, (X, y) in enumerate(it.izip(self.Xs, self.ys)):
                preds = get_predictions(classifiers, features, X)
                classes = get_classes(weight_set, preds)
                acc = accuracy_score(y, classes)
                accs[counter] = acc
                counter += 1

        output = os.path.join(self.output_path, self.date + '_' + 'report_' + func.__name__ + '.csv')
        # if not created
        pth = Path(output)
        if not pth.exists():
            with open(output, 'w') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow(['method', 'fold', 'run', 'generation', 'individual', 'set_name', 'set_size', 'accuracy'])

        counter = 0
        with open(output, 'a') as f:
            writer = csv.writer(f, delimiter=',')

            for i in xrange(n_individuals):
                for j in xrange(n_sets):
                    writer.writerow([func.__name__, self.fold, self.run, str(gen), str(i), self.set_names[j], self.set_sizes[j], str(accs[counter])])
                    counter += 1

        lock.release()

    def __save_population__(self, output_path, date, func, population, lock):
        """

        :param output_path:
        :param date:
        :param func:
        :param population:
        :type lock: multiprocessing.Lock
        :param lock:
        :return:
        """

        lock.acquire()

        if func.__name__ == generate.__name__:
            dense = np.array(map(lambda x: x.tolist(), population))
        elif func.__name__ == integrate.__name__:
            dense = np.array(map(lambda x: x.ravel(), population))
        else:
            dense = population

        pd.DataFrame(dense).to_csv(
            os.path.join(output_path, date + '_' + 'population' + '_' + func.__name__ + '.csv'),
            sep=',',
            index=False,
            header=False
        )

        lock.release()

    def join_all(self):
        """
        Join all running processes.
        """

        for p in self.processes:
            p.join()


def eel(params, X_train, y_train, X_val, y_val, X_test, y_test, reporter=None):
    print '-------------------------------------------------------'
    print '--------------------- generation ----------------------'
    print '-------------------------------------------------------'

    classifiers, features, fitness = generate(
        X_train, y_train, X_val, y_val,
        base_classifier=clf,
        n_classifiers=params['generation']['n_individuals'],
        n_generations=params['generation']['n_generations'],
        reporter=reporter
    )

    val_predictions = get_predictions(classifiers, features, X_val)
    test_predictions = get_predictions(classifiers, features, X_test)

    print '-------------------------------------------------------'
    print '---------------------- selection ----------------------'
    print '-------------------------------------------------------'

    if params['selection']['n_generations'] == 0:
        best_classifiers = np.ones(len(classifiers), dtype=np.bool)
    else:
        best_classifiers = eda_select(
            features, classifiers, val_predictions, y_val,
            n_individuals=params['selection']['n_individuals'],
            n_generations=params['selection']['n_generations'],
            reporter=reporter
        )

    print '-------------------------------------------------------'
    print '--------------------- integration ---------------------'
    print '-------------------------------------------------------'

    if params['integration']['n_generations'] == 0:
        _best_weights = np.ones((len(best_classifiers), len(np.unique(y_val))), dtype=np.float32)
    else:
        _best_weights = integrate(
            features[np.where(best_classifiers)], classifiers[np.where(best_classifiers)],
            val_predictions[np.where(best_classifiers)], y_val,
            n_individuals=params['integration']['n_individuals'],
            n_generations=params['integration']['n_generations'],
            reporter=reporter
        )

    '''
        Now testing
    '''

    y_test_pred = get_classes(_best_weights, test_predictions[np.where(best_classifiers)])
    return y_test_pred


if __name__ == '__main__':
    params = json.load(open('../params.json', 'r'))

    metaparams = params['metaparams']

    X_train, y_train, X_val, y_val, X_test, y_test = path_to_sets(
        params['full_path'],
        train_size=0.5,
        test_size=0.25,
        val_size=0.25,
        random_state=params['random_state']
    )

    reporter = Reporter(
        Xs=[X_train, X_val, X_test],
        ys=[y_train, y_val, y_test],
        set_names=['train', 'val', 'test'],
        output_path=params['reporter_output'],
        fold=1
    )

    preds = eel(metaparams, X_train, y_train, X_val, y_val, X_test, y_test, reporter=reporter)
    acc = accuracy_score(y_test, preds)
    print 'test accuracy: %.2f' % acc

    reporter.join_all()
