"""
This is the main script for running the baseline algorithms reported in the paper.
It should output a folder full of metadata, as well as the summary of results for each algorithm.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import pathlib2
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold

from linear import LogisticAdaBoost, AdaBoostOnes, AdaBoostNormal
from utils import path_to_dataframe
from reporter import BaselineReporter

__author__ = 'Henry Cagnini'


def test_baselines(datasets_path, output_path, params_path):
    """
    Runs baselines.


    :type datasets_path: str
    :param datasets_path: path to a folder where datasets will be used for tests.
    :type output_path: str
    :param output_path: path to output metadata regarding evolutionary process.
    :type params_path: str
    :param params_path: path to parameters file.
    """

    n_runs = 10

    params = json.load(open(params_path, 'r'))
    datasets = [str(xx).split('/')[-1] for xx in pathlib2.Path(datasets_path).iterdir() if xx.is_file()]
    algorithms = [LogisticAdaBoost, AdaBoostClassifier, AdaBoostOnes, AdaBoostNormal]

    for dataset in datasets:
        dataset_name = dataset.split('/')[-1].split('.')[-2]

        print ('testing %s dataset' % dataset_name)

        full_df = path_to_dataframe(os.path.join(datasets_path, dataset))

        y_name = full_df.columns[-1]

        full_df[y_name] = pd.Categorical(full_df[y_name])
        full_df[y_name] = full_df[y_name].cat.codes

        X = full_df[full_df.columns[:-1]]
        y = full_df[full_df.columns[-1]]

        skf = StratifiedKFold(n_splits=params['n_folds'], shuffle=True, random_state=params['random_state'])

        for n_fold, (train_index, test_index) in enumerate(skf.split(X, y)):

            X_train = X.iloc[train_index]
            X_test = X.iloc[test_index]

            y_train = y.iloc[train_index]
            y_test = y.iloc[test_index]

            n_classes = len(np.unique(y_train))

            for n_run in xrange(n_runs):
                for algorithm in algorithms:
                    print ('# --- dataset: %r n_fold: %r n_run: %r algorithm: %r --- #' % (
                        dataset_name, n_fold, n_run, algorithm.__name__
                    ))

                    reporter = BaselineReporter(
                        Xs=[X_train, X_test],
                        ys=[y_train, y_test],
                        set_names=['train', 'test'],
                        output_path=output_path,
                        dataset_name=dataset_name,
                        n_fold=n_fold,
                        n_run=n_run,
                        n_classifiers=params['n_base_classifiers'],
                        n_classes=n_classes,
                        algorithm=algorithm
                    )

                    if algorithm.__name__ == 'AdaBoostClassifier':
                        inst = algorithm(n_estimators=params['n_base_classifiers'], algorithm='SAMME').fit(
                            X_train, y_train
                        )
                    else:
                        inst = algorithm(n_estimators=params['n_base_classifiers']).fit(X_train, y_train)

                    reporter.save_baseline(inst)


if __name__ == '__main__':
    if len(sys.argv) != 5:
        print ('usage:')
        print ('\tpython test_baselines.py <path_datasets> <path_metadata> <path_params> <path_results>')
        print ('example:')
        print ('\tpython test_baselines.py \"/home/user/datasets\" \"/home/user/metadata\"' + \
              '\"/home/user/params.json\" \"/home/user/results.csv\"')
    else:
        __dataset_path, __output_path, __params_path, __results_path = sys.argv[1:]
        test_baselines(__dataset_path, __output_path, __params_path)
        BaselineReporter.generate_summary(__output_path, __results_path)
