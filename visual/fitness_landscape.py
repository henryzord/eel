"""
Script for generating fitness landscape for a single run of EEL.

It generates both a static and animated fitness landscape:

static: "A photograph" of the fitness landscape throughout evolution. Each cell contains the mean weight
for that spot in the landscape.
animated: A gif that demonstrates the evolution of the individuals' fitness throughout evolution. Each cell
contains the mean weight for that spot in the landscape.
"""

import matplotlib.animation as animation
import pandas as pd
import pathlib2
from matplotlib.pylab import *
import os


def animated_projection(table, file_out):
    inner_table = table.loc[(table['set_name'] == 'train') & (table['generation'] >= 0)]
    weight_columns = inner_table.columns[18:]

    x_columns = weight_columns[:(len(weight_columns) / 2)]
    y_columns = weight_columns[(len(weight_columns) / 2):]

    fig, ax = plt.subplots(1)

    min_fitness, max_fitness = inner_table['fitness'].min(), inner_table['fitness'].max()

    def __update_data__(curr):
        partial = inner_table.loc[inner_table['generation'] <= curr]
        x = np.mean(partial[x_columns].values, axis=1)
        y = np.mean(partial[y_columns].values, axis=1)
        c = partial['fitness'].values

        ax.clear()

        ax.set_xlim(0.7, 1.)
        ax.set_ylim(0.7, 1.)

        ax.hexbin(x, y, C=c, reduce_C_function=np.mean, vmin=min_fitness, vmax=max_fitness)
        ax.set_xlabel(r'Mean of first W/2 weights')
        ax.set_ylabel(r'Mean of last W/2 weights')

    simulation = animation.FuncAnimation(
        fig, __update_data__, frames=np.arange(inner_table['generation'].max()), interval=150, repeat=True
    )
    simulation.save(file_out, dpi=80, writer='imagemagick')

    plt.close()


def static_projection(table, file_out):
    plt.figure()

    inner_table = table.loc[(table['set_name'] == 'train') & (table['generation'] >= 0)]
    weight_columns = inner_table.columns[18:]

    min_fitness, max_fitness = inner_table['fitness'].min(), inner_table['fitness'].max()
    
    x_columns = weight_columns[:(len(weight_columns) / 2)]
    y_columns = weight_columns[(len(weight_columns) / 2):]

    x = np.mean(inner_table[x_columns].values, axis=1)
    y = np.mean(inner_table[y_columns].values, axis=1)
    c = inner_table['fitness'].values

    plt.xlim(0.7, 1.)
    plt.ylim(0.7, 1.)

    plt.hexbin(x, y, C=c, reduce_C_function=np.mean, vmin=min_fitness, vmax=max_fitness)  # mean of weights

    plt.xlabel(r'Mean of first W/2 weights')
    plt.ylabel(r'Mean of last W/2 weights')

    plt.colorbar()
    plt.savefig(file_out, type='pdf')
    plt.close()


def main(path_read, path_out):
    files = [x for x in pathlib2.Path(path_read).iterdir() if (x.is_file() and 'pop.csv' in str(x))]
    files = map(lambda xx: str(xx).split('/')[-1].split('.')[0].split('-'), files)
    summary = pd.DataFrame(files, columns=['dataset_name', 'n_fold', 'n_run', 'pop'])
    summary['n_fold'] = summary['n_fold'].astype(np.int32)
    summary['n_run'] = summary['n_run'].astype(np.int32)

    datasets = summary['dataset_name'].unique()
    n_folds = len(summary['n_fold'].unique())
    n_runs = len(summary['n_run'].unique())

    string_columns = ['dataset', 'set_name']

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
                        '-'.join(
                            [dataset_name, str(n_fold), str(n_run), 'pop']
                        ) + '.csv',
                    ),
                    sep=','
                )

                if numeric_columns is None:
                    numeric_columns = current.columns.difference(string_columns)
                current[numeric_columns] = current[numeric_columns].astype(np.float32)

                static_projection(
                    current, os.path.join(
                        path_out, '-'.join([dataset_name, str(n_fold), str(n_run), 'static_landscape']) + '.pdf'
                    )
                )

                animated_projection(
                    current, os.path.join(
                        path_out, '-'.join([dataset_name, str(n_fold), str(n_run), 'dynamic_landscape']) + '.gif'
                    )
                )

                global_counter += 1
                print '%04.d/%04.d steps done' % (global_counter, total_steps)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print 'usage:'
        print '\tpython fitness_landscape.py <path_read> <path_out>'
        print 'example:'
        print '\tpython fitness_landscape.py \"/home/user/metadata\" \"home/user/result.csv\"'

    __path_read, __path_out = sys.argv[1:]

    main(__path_read, __path_out)
