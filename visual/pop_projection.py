"""
Generates graphical model projections for a single run of EEL.
"""

import os
import sys

import numpy as np
import pandas as pd
import pathlib2
from matplotlib import cm, pyplot as plt
from matplotlib.colors import to_hex


def pop_projection(table, file_out):
    table.reset_index(inplace=True)
    table.fillna(0, inplace=True)

    metrics = ['accuracy', 'precision-micro', 'recall-micro', 'f1-micro', 'fitness']
    line_colors = map(to_hex, cm.viridis(np.linspace(0, 1, len(metrics) + 1)[:-1]))

    f, axarr = plt.subplots(nrows=len(metrics), sharex=True, figsize=(5, 5), dpi=166)
    axarr[0].set_title('training set statistics')

    table = table.loc[table['set_name'] == 'train']

    x = table['generation'] + 1  # sums 1 so prior generation is 0

    for i, metric in enumerate(metrics):
        if metric == 'fitness':
            x = x[1:]
            y_mean = table[(metric, 'mean')][1:]
            y_std = table[(metric, 'std')][1:]
        else:
            y_mean = table[(metric, 'mean')]
            y_std = table[(metric, 'std')]

        axarr[i].set_xlim(min(x), max(x))
        axarr[i].plot(x, y_mean, c=line_colors[i], linestyle='-', label=metric.split('-')[0])
        axarr[i].fill_between(x, y_mean - y_std, y_mean + y_std, color=line_colors[i], alpha=0.5, label=None)
        axarr[i].set_ylabel(metric.split('-')[0])
        axarr[i].yaxis.set_label_position("right")

        lower_tick = round(min(y_mean - y_std), 2)
        upper_tick = round(max(y_mean + y_std), 2)
        ticks = np.linspace(lower_tick, upper_tick, 3)
        tick_labels = map(lambda k: '%.2f' % k, ticks)

        axarr[i].set_yticks(ticks)
        axarr[i].set_yticklabels(tick_labels)

    plt.xlabel('Generation')

    plt.savefig(file_out, type='pdf')
    plt.close()


def main(path_read, path_out):
    files = [xx for xx in pathlib2.Path(path_read).iterdir() if (xx.is_file() and 'pop.csv' in str(xx))]
    files = map(lambda x: str(x).split('/')[-1].split('.')[0].split('-'), files)
    summary = pd.DataFrame(files, columns=['dataset_name', 'n_fold', 'n_run', 'pop'])
    summary['n_fold'] = summary['n_fold'].astype(np.int32)
    summary['n_run'] = summary['n_run'].astype(np.int32)

    datasets = summary['dataset_name'].unique()
    n_folds = len(summary['n_fold'].unique())
    n_runs = len(summary['n_run'].unique())

    string_columns = ['dataset', 'set_name']

    # metric_names = [metric_name for metric_name, metric in Reporter.metrics]

    total_steps = len(datasets) * n_folds * n_runs
    global_counter = 0
    for dataset_name in datasets:
        # checks whether required dataset exists
        partial = summary.loc[summary['dataset_name'] == dataset_name]
        if len(partial.index) != (n_folds * n_runs):
            global_counter += (n_folds * n_runs)
            print '%04.d/%04.d steps done [skipping %s]' % (global_counter, total_steps, dataset_name)
            continue  # skips

        numeric_columns = None
        for n_fold in xrange(n_folds):
            for n_run in xrange(n_runs):
                current = pd.read_csv(
                    os.path.join(
                        path_read,
                        '-'.join([dataset_name, str(n_fold), str(n_run), 'pop']) + '.csv',
                    ),
                    sep=','
                )

                if numeric_columns is None:
                    numeric_columns = current.columns.difference(string_columns)
                current[numeric_columns] = current[numeric_columns].astype(np.float32)

                gb = current.groupby(by=['generation', 'set_name'])
                res = gb.agg([np.mean, np.std, np.median])

                pop_projection(
                    res, os.path.join(path_out, '-'.join([dataset_name, str(n_fold), str(n_run), 'pop_plot']) + '.pdf')
                )

                global_counter += 1
                print '%04.d/%04.d steps done' % (global_counter, total_steps)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print 'usage:'
        print '\tpython pop_projection.py <path_read> <path_out>'
        print 'example:'
        print '\tpython pop_projection.py \"/home/user/metadata\" \"home/user/result.csv\"'

    __path_read, __path_out = sys.argv[1:]

    main(__path_read, __path_out)
