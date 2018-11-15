"""
Runs a single instance of EEL.
If an exception occurs during this execution, the program will output the exception message to an file, into the
output_path directory.

Command line parameters:
    * path_datasets: a full path (including file type, i.e. dataset.arff) to the dataset to be run.
    * path_metadata: path to output metadata regarding evolutionary process.
    * path_params: path to parameters file.
    * n_fold: number of fold to run in this execution. Must be smaller than the value in the params file.
    * n_run: number of the current run. Note: it is not the total amount of runs!
"""
import argparse
import json

import numpy as np

from eda import Ensemble
from eda.integration import integrate
from reporter import EDAReporter
from utils import __get_fold__, get_dataset_name
from data_normalization import DataNormalizer


def eelem(dataset_path, output_path, params_path, n_fold, n_run, verbose=True):
    """
    Runs a single instance of EEL.


    :type dataset_path: str
    :param dataset_path: a full path (including file type, i.e. dataset.arff) to the dataset to be run.
    :type output_path: str
    :param output_path: path to output metadata regarding evolutionary process.
    :type params_path: str
    :param params_path: path to parameters file.
    :type n_fold: int
    :param n_fold: number of fold to run in this execution. Must be smaller than the value in the params file.
    :type n_run: int
    :param n_run: number of the current run.
    :type verbose: bool
    :param verbose: whether to output metadata to console. Defaults to True.
    """

    params = json.load(open(params_path))

    dataset_name = get_dataset_name(dataset_path)

    X_train, X_test, y_train, y_test = __get_fold__(params=params, dataset_path=dataset_path, n_fold=n_fold)

    n_classes = len(np.unique(y_train))

    reporter = EDAReporter(
        Xs=[X_train, X_test],
        ys=[y_train, y_test],
        set_names=['train', 'test'],
        output_path=output_path,
        dataset_name=dataset_name,
        n_fold=n_fold,
        n_run=n_run,
        n_classifiers=params['n_base_classifiers'],
        n_classes=n_classes,
    )

    ensemble = Ensemble.from_adaboost(
        X_train=X_train, y_train=y_train,
        data_normalizer_class=DataNormalizer,
        n_classifiers=params['n_base_classifiers'],
    )  # type: Ensemble

    ensemble = integrate(
        ensemble=ensemble,
        n_individuals=params['n_individuals'],
        n_generations=params['n_generations'],
        reporter=reporter,
        verbose=verbose
    )
    return ensemble


def preliminaries(dataset_path, output_path, params_path, n_fold, n_run):
    """
    Runs a single instance of EEL.
    If an exception occurs during this execution, the program will output the exception message to an file, into the
    output_path directory.

    :type dataset_path: str
    :param dataset_path: a full path (including file type, i.e. dataset.arff) to the dataset to be run.
    :type output_path: str
    :param output_path: path to output metadata regarding evolutionary process.
    :type params_path: str
    :param params_path: path to parameters file.
    :type n_fold: int
    :param n_fold: number of fold to run in this execution. Must be smaller than the value in the params file.
    :type n_run: int
    :param n_run: number of the current run.
    """

    dataset_name = get_dataset_name(dataset_path)

    # try:
    eelem(
        dataset_path=dataset_path,
        output_path=output_path,
        params_path=params_path,
        n_fold=n_fold,
        n_run=n_run
    )
    # TODO reactivate later
    # except Exception as e:
    #     name = EDAReporter.get_output_file_name(
    #         output_path=output_path,
    #         dataset_name=dataset_name,
    #         n_fold=n_fold, n_run=n_run,
    #         reason='exception'
    #     )
    #
    #     with open(name, 'w') as f:
    #         f.write(str(e) + '\n' + str(e.args))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Main script for running Estimation of Distribution Algorithms for ensemble learning.'
    )
    parser.add_argument(
        '-d', action='store', required=True,
        help='Path to datasets folder. Datasets must be in .arff format.'
    )
    parser.add_argument(
        '-m', action='store', required=True,
        help='Path to metadata folder. The folder must be pre-existent, even if empty.'
    )
    parser.add_argument(
        '-p', action='store', required=True,
        help='Path to EEL\'s .json parameter file.'
    )
    parser.add_argument(
        '--n_fold', action='store', required=True, type=int,
        help='Index of the fold currently being tested.'
    )
    parser.add_argument(
        '--n_run', action='store', required=True, type=int,
        help='Index of the run currently being tested.'
    )

    args = parser.parse_args()
    preliminaries(args.d, args.m, args.p, args.n_fold, args.n_run)
