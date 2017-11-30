import itertools as it
import os

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as off
from matplotlib import cm


def __plot__(data, df):
    gb = df.groupby(by=['method', 'generation']).aggregate([np.mean])

    gb.reset_index(inplace=True)

    generation = gb['generation']

    _data = gb.drop(['generation', 'method', 'fold', 'run'], axis=1).values

    data['data'] = _data
    data['generation'] = generation
    return data


def main():
    off.init_notebook_mode()
    output_name = 'gm.html'
    path_read = '/home/henry/Projects/eel/metadata'

    generate_df = pd.DataFrame([])
    select_df = pd.DataFrame([])
    integrate_df = pd.DataFrame([])

    files = os.listdir(path_read)
    for _file in files:
        if '_gm_integrate' in _file:
            integrate_df = pd.read_csv(os.path.join(path_read, _file), sep=',')
        elif '_gm_generate' in _file:
            generate_df = pd.read_csv(os.path.join(path_read, _file), sep=',')
        elif '_gm_select' in _file:
            select_df = pd.read_csv(os.path.join(path_read, _file), sep=',')

    step_names = filter(
        None,
        [bool(len(generate_df) > 0) * 'generate',
         bool(len(select_df) > 0) * 'select',
         bool(len(integrate_df) > 0) * 'integrate']
    )

    overall = {
        step_name: dict() for step_name in ['generate', 'select', 'integrate']
    }

    if 'generate' in step_names:
        overall['generate'] = __plot__(overall['generate'], generate_df)
    else:
        del overall['generate']
    if 'select' in step_names:
        overall['select'] = __plot__(overall['select'], select_df)
    else:
        del overall['select']
    if 'integrate' in step_names:
        overall['integrate'] = __plot__(overall['integrate'], integrate_df)
    else:
        del overall['integrate']

    traces = []
    for step_name in overall.keys():
        traces += [
            go.Heatmap(
                z=overall[step_name]['data'],
                colorscale='Magma',
                zauto=False,
                zmax=1.,
                zmin=0.,
                colorbar=dict(
                    title='certainty'
                )
            )
        ]

    layout = dict(
        title='Graphical models',
        xaxis=dict(
            title='Variable',
        ),
        yaxis=dict(
            title='Generation'
        )
    )

    fig = go.Figure(
        data=traces,
        layout=layout
    )
    off.plot(fig, filename=output_name)


if __name__ == '__main__':
    main()
