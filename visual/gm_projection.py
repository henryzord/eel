"""
Generates graphical model projections for a single run of EEL.
"""

import os
import sys

import numpy as np
import pandas as pd
import pathlib2
from matplotlib import pyplot as plt


def gm_projection(table, file_out):
    plt.figure()

    weight_columns = table.columns[5:]

    n_generations, n_weights = table[weight_columns].shape

    plt.imshow(table[weight_columns], cmap='viridis', interpolation='nearest', origin='lower', aspect='auto')
    plt.ylabel('Generation')
    plt.xlabel('Weights')

    y_ticks = np.linspace(0, n_generations - 1, n_generations / 4).astype(np.int32)
    y_tick_labels = map(str, y_ticks)

    plt.yticks(y_ticks, y_tick_labels)

    x_ticks = np.linspace(start=0, stop=n_weights - 1, num=5, endpoint=True, dtype=np.int32)
    x_tick_labels = map(
        lambda x: r'$%s_{%s,%s}$' % tuple(x.split('_')),
        weight_columns[x_ticks]
    )

    plt.xticks(
        x_ticks,
        x_tick_labels
    )

    plt.colorbar()

    plt.tight_layout()
    plt.savefig(file_out, type='pdf')
    plt.close()


def main(path_read, path_out):
    files = [xx for xx in pathlib2.Path(path_read).iterdir() if (xx.is_file() and 'gm.csv' in str(xx))]
    files = map(lambda x: str(x).split('/')[-1].split('.')[0].split('-'), files)
    summary = pd.DataFrame(files, columns=['dataset_name', 'n_fold', 'n_run', 'pop'])
    summary['n_fold'] = summary['n_fold'].astype(np.int32)
    summary['n_run'] = summary['n_run'].astype(np.int32)

    datasets = summary['dataset_name'].unique()
    n_folds = len(summary['n_fold'].unique())
    n_runs = len(summary['n_run'].unique())

    string_columns = ['dataset']

    total_steps = len(datasets) * n_folds * n_runs
    global_counter = 0
    for dataset_name in datasets:
        # checks whether required dataset exists
        partial = summary.loc[summary['dataset_name'] == dataset_name]
        if len(partial.index) != (n_folds * n_runs):
            global_counter += (n_folds * n_runs)
            print '%04.d/%04.d steps done [skipping %s]' % (
                global_counter, total_steps, dataset_name
            )
            continue  # skips

        numeric_columns = None
        for n_fold in xrange(n_folds):
            for n_run in xrange(n_runs):
                current = pd.read_csv(
                    os.path.join(
                        path_read,
                        '-'.join([dataset_name, str(n_fold), str(n_run), 'gm']) +
                        '.csv',
                    ),
                    sep=','
                )

                if numeric_columns is None:
                    numeric_columns = current.columns.difference(string_columns)
                current[numeric_columns] = current[numeric_columns].astype(np.float32)

                gm_projection(
                    current, os.path.join(
                        path_out, '-'.join([dataset_name, str(n_fold), str(n_run), 'gm_plot']) + '.pdf'
                    )
                )

                global_counter += 1
                print '%04.d/%04.d steps done' % (global_counter, total_steps)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print 'usage:'
        print '\tpython gm_projection.py <path_read> <path_out>'
        print 'example:'
        print '\tpython gm_projection.py \"/home/user/metadata\" \"home/user/result.csv\"'

    __path_read, __path_out = sys.argv[1:]

    main(__path_read, __path_out)
