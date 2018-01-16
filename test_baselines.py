"""
Tests several datasets and store the results.
"""

import json
import os
from datetime import datetime as dt

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from utils import path_to_dataframe

__author__ = 'Henry Cagnini'


def main():
    datasets_path = 'datasets/uci'
    params_path = 'params.json'
    output_path = "metadata"

    datasets = os.listdir(datasets_path)
    params = json.load(open(params_path))

    for dataset in datasets:
        dataset_name = dataset.split('.')[0]

        print 'testing %s dataset' % dataset_name

        full_df = path_to_dataframe(datasets_path)

        y_name = full_df.columns[-1]

        full_df[y_name] = pd.Categorical(full_df[y_name])
        full_df[y_name] = full_df[y_name].cat.codes

        X = full_df[full_df.columns[:-1]]
        y = full_df[full_df.columns[-1]]

        acc_randomforest = []
        acc_adaboost = []
        acc_xgb = []

        std_randomforest = []
        std_adaboost = []
        std_xgb = []

        n_all = X.shape[0]

        # variables = {}
        # sys.stdout = open(
        #     os.path.join(params_file['reporter_output'], dataset_name + '.txt'), 'w'
        # )

        skf = StratifiedKFold(n_splits=params['n_folds'], shuffle=True, random_state=params['random_state'])

        date = dt.now()

        for fold, (train_index, test_index) in enumerate(skf.split(X, y)):

            _run_adaboost = []
            _run_randomforest = []
            _run_xgb = []

            for run in xrange(params['n_runs']):
                X_train_val = X.iloc[train_index]
                X_test = X.iloc[test_index]

                y_train_val = y.iloc[train_index]
                y_test = y.iloc[test_index]

                n_test = X_test.shape[0]

                adaboost = AdaBoostClassifier(
                    n_estimators=params['n_base_classifiers']).fit(
                    X_train_val, y_train_val)  # type: AdaBoostClassifier
                randomforest = RandomForestClassifier(
                    n_estimators=params['n_base_classifiers']
                ).fit(X_train_val, y_train_val)  # type: RandomForestClassifier
                xgb = XGBClassifier(
                    n_estimators=params['n_base_classifiers']
                ).fit(X_train_val, y_train_val)

                preds_adaboost = adaboost.predict(X_test)
                preds_randomforest = randomforest.predict(X_test)
                preds_xgb = xgb.predict(X_test)

                __acc_adaboost = accuracy_score(y_test, preds_adaboost)
                __acc_randomforest = accuracy_score(y_test, preds_randomforest)
                __acc_xgb = accuracy_score(y_test, preds_xgb)

                _run_adaboost += [__acc_adaboost * (float(n_test) / n_all)]  # accuracy for that run
                _run_randomforest += [__acc_randomforest * (float(n_test) / n_all)]  # accuracy for that run
                _run_xgb += [__acc_xgb * (float(n_test) / n_all)]  # accuracy for that run

                print '------ run accuracies: -----'
                print '\tadaboost run accuracy: %.4f' % _run_adaboost[-1]
                print '\trandomForest run accuracy: %.4f' % _run_randomforest[-1]
                print '\txgb run accuracy: %.4f' % _run_xgb[-1]
                print '------------------------------'

                # -------- accuracy for that fold -------- #
            acc_adaboost += [np.mean(_run_adaboost)]
            acc_randomforest += [np.mean(_run_randomforest)]
            acc_xgb += [np.mean(_run_xgb)]

            std_adaboost += [np.std(_run_adaboost)]
            std_randomforest += [np.std(_run_randomforest)]
            std_xgb += [np.std(_run_xgb)]

            print '----------------------------------'
            print '------ partition accuracies: -----'
            print '----------------------------------'
            print '\tadaboost accuracy: %.4f +- %.4f' % (acc_adaboost[-1], std_adaboost[-1])
            print '\trandomForest accuracy: %.4f +- %.4f' % (acc_randomforest[-1], std_randomforest[-1])
            print '\txgb accuracy: %.4f +- %.4f' % (acc_xgb[-1], std_xgb[-1])
            print '----------------------------------'
            print '----------------------------------'

        print 'adaboost accuracy: %.4f +- %.4f' % (sum(acc_adaboost), np.mean(std_adaboost))
        print 'randomForest accuracy: %.4f +- %.4f' % (sum(acc_randomforest), np.mean(std_randomforest))
        print 'xgb accuracy: %.4f +- %.4f' % (sum(acc_xgb), np.mean(std_xgb))

        # try:
        #     execfile('__main__.py', variables)
        # except Exception as e:
        #     with open(os.path.join(params['reporter_output'], 'exception.txt'), 'w') as f:
        #         f.write(str(e.message) + '\n' + str(e.args))


if __name__ == '__main__':
    main()
