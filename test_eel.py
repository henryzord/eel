"""
This is the main script for running EEL.
It should output a folder full of metadata, as well as the summary of results for EEL.
"""

import json
import os
import sys
import time
from multiprocessing import Process
from subprocess import Popen

import pathlib2

from main import preliminaries
from reporter import EDAReporter

__author__ = 'Henry Cagnini'


def __get_running_processes__(jobs):
    for job in jobs:  # type: Process
        if not job.is_alive():
            jobs.remove(job)
    return jobs


def __get_running_jobs__(jobs):

    for job in jobs:  # type: Popen
        poll = job.poll()
        if poll is not None:  # subprocess is alive
            jobs.remove(job)

    return jobs


def test_eel(datasets_path, output_path, params_path, n_runs, n_jobs):
    """
    Runs several instances of EEL, generating metadata along the process.

    :type datasets_path: str
    :param datasets_path: path to a folder where datasets will be used for tests.
    :type output_path: str
    :param output_path: path to output metadata regarding evolutionary process.
    :type params_path: str
    :param params_path: path to parameters file.
    :type n_runs: int
    :param n_runs: number of runs to test.
    :type n_jobs: int
    :param n_jobs: number of parallel processes to use for running tests.
    """

    params = json.load(open(params_path, 'r'))

    datasets = [str(xx).split('/')[-1] for xx in pathlib2.Path(datasets_path).iterdir() if xx.is_file()]

    jobs = []

    for dataset in datasets:
        dataset_name = dataset.split('/')[-1].split('.')[-2]

        print 'testing %s dataset' % dataset_name

        for n_fold in xrange(params['n_folds']):
            for n_run in xrange(n_runs):
                print '# --- dataset: %r n_fold: %r/%r n_run: %r/%r --- #' % (
                    dataset_name, n_fold + 1, params['n_folds'], n_run + 1, n_runs
                )

                while len(jobs) >= n_jobs:
                    jobs = __get_running_processes__(jobs)
                    time.sleep(5)

                job = Process(
                    target=preliminaries,
                    kwargs=dict(
                        dataset_path=os.path.join(datasets_path, dataset),
                        output_path=output_path,
                        params_path=params_path,
                        n_fold=n_fold,
                        n_run=n_run,
                    )
                )
                job.start()
                jobs += [job]

    while len(jobs) > 0:
        jobs = __get_running_processes__(jobs)
        time.sleep(5)


if __name__ == '__main__':
    if len(sys.argv) != 7:
        print 'usage:'
        print '\tpython test_eel.py <path_datasets> <path_metadata> <path_params> <path_results> <n_runs> <n_jobs>'
        print 'example:'
        print '\tpython test_eel.py \"/home/user/datasets\" \"/home/user/metadata\"' + \
              '\"/home/user/params.json\" \"/home/user/results.csv\" 10 4'

    else:
        __datasets_path, __output_path, __params_path, __results_path, __n_runs, __n_jobs = sys.argv[1:]
        test_eel(
            __datasets_path,
            __output_path,
            __params_path,
            int(__n_runs),
            int(__n_jobs)
        )
        EDAReporter.generate_summary(__output_path, __results_path)
