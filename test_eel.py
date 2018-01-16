"""
Tests several datasets and store the results.
"""

import json
import os
from datetime import datetime as dt

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from eda import Ensemble
from eda import Reporter
from eda.integration import integrate
from utils import path_to_dataframe

__author__ = 'Henry Cagnini'


def eelem(params, X_train, y_train, use_weights=False, reporter=None):
    t1 = dt.now()

    ensemble = Ensemble.from_adaboost(
        X_train=X_train, y_train=y_train,
        n_classifiers=params['n_base_classifiers'],
        n_generations=params['n_generations'],
        use_weights=use_weights,
    )  # type: Ensemble

    ensemble_train_acc = accuracy_score(ensemble.y_train, ensemble.predict(ensemble.X_train))
    dfd = ensemble.dfd(ensemble.X_train, ensemble.y_train)

    print 'generation 0: ens val acc: %.4f dfd: %.4f time elapsed: %f' % (
        ensemble_train_acc, dfd, (dt.now() - t1).total_seconds()
    )

    ensemble = integrate(
        ensemble=ensemble,
        n_individuals=params['n_individuals'],
        n_generations=params['n_generations'],
        reporter=reporter
    )
    return ensemble


def main():
    datasets_path = 'datasets/uci'
    params_path = 'params.json'
    output_path = "metadata"

    datasets = os.listdir(datasets_path)
    params = json.load(open(params_path))

    for dataset in datasets:
        dataset_name = dataset.split('.')[0]

        print 'testing %s dataset' % dataset_name

        full_df = path_to_dataframe(os.path.join(datasets_path, dataset))

        y_name = full_df.columns[-1]

        full_df[y_name] = pd.Categorical(full_df[y_name])
        full_df[y_name] = full_df[y_name].cat.codes

        X = full_df[full_df.columns[:-1]]
        y = full_df[full_df.columns[-1]]

        acc_eelem = []
        std_eelem = []

        n_all = X.shape[0]

        # variables = {}
        # sys.stdout = open(
        #     os.path.join(params_file['reporter_output'], dataset_name + '.txt'), 'w'
        # )

        skf = StratifiedKFold(n_splits=params['n_folds'], shuffle=True, random_state=params['random_state'])

        alias = params['alias']

        for fold, (train_index, test_index) in enumerate(skf.split(X, y)):

            _run_eelem = []

            for run in xrange(params['n_runs']):
                X_train = X.iloc[train_index]
                X_test = X.iloc[test_index]

                y_train = y.iloc[train_index]
                y_test = y.iloc[test_index]

                reporter = Reporter(
                    Xs=[X_train, X_test],
                    ys=[y_train, y_test],
                    set_names=['train', 'test'],
                    output_path=output_path,
                    alias=alias,
                    fold=fold,
                    n_run=run
                )

                n_test = X_test.shape[0]

                ensemble = eelem(params, X_train, y_train, use_weights=params['use_weights'], reporter=reporter)
                preds_eel = ensemble.predict(X_test)

                __acc_eelem = accuracy_score(y_test, preds_eel)
                _run_eelem += [__acc_eelem * (float(n_test) / n_all)]  # accuracy for that run

                print '------ run accuracies: -----'
                print '\teelem run accuracy: %.4f' % _run_eelem[-1]
                print '------------------------------'

                # -------- accuracy for that fold -------- #
            acc_eelem += [np.mean(_run_eelem)]  # the accuracy for eelem in that fold is the mean for N runs
            std_eelem += [np.std(_run_eelem)]

            print '----------------------------------'
            print '------ partition accuracies: -----'
            print '----------------------------------'
            print '\teelem accuracy: %.4f +- %.4f' % (acc_eelem[-1], std_eelem[-1])
            print '----------------------------------'
            print '----------------------------------'

        print 'eelem accuracy: %.4f +- %.4f' % (sum(acc_eelem), np.mean(std_eelem))

        # try:
        #     execfile('__main__.py', variables)
        # except Exception as e:
        #     with open(os.path.join(params['reporter_output'], 'exception.txt'), 'w') as f:
        #         f.write(str(e.message) + '\n' + str(e.args))


if __name__ == '__main__':
    main()
