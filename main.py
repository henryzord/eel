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

import json
import sys

import numpy as np

from eda import Ensemble
from reporter import EDAReporter
from eda.integration import integrate
from utils import __get_fold__, get_dataset_name


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

    try:
        eelem(
            dataset_path=dataset_path,
            output_path=output_path,
            params_path=params_path,
            n_fold=n_fold,
            n_run=n_run
        )

    except Exception as e:
        name = EDAReporter.get_output_file_name(
            output_path=output_path,
            dataset_name=dataset_name,
            n_fold=n_fold, n_run=n_run,
            reason='exception'
        )

        with open(name, 'w') as f:
            f.write(str(e.message) + '\n' + str(e.args))


if __name__ == '__main__':
    if len(sys.argv) != 6:
        print ('usage:')
        print ('\tpython main.py <path_datasets> <path_metadata> <path_params> <n_fold> <n_run>')
        print ('example:')
        print ('\tpython main.py \"/home/user/datasets\" \"/home/user/metadata\"' + \
              '\"/home/user/params.json\" 0 0')

    else:
        __dataset_path, __output_path, __params_path, __n_fold, __n_run = sys.argv[1:]

        __dataset_path = str(__dataset_path)
        __output_path = str(__output_path)
        __params_path = str(__params_path)
        __n_fold = int(__n_fold)
        __n_run = int(__n_run)

        preliminaries(__dataset_path, __output_path, __params_path, __n_fold, __n_run)
