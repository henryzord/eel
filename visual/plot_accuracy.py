import itertools as it
import os

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as off
from matplotlib import cm


def __plot__(data, df):
    set_names = df['set_name'].unique()

    for i, set_name in enumerate(set_names):
        _subset = df.loc[df['set_name'] == set_name]
        gb = _subset.groupby(by=['method', 'generation', 'set_name', 'set_size']).aggregate([np.mean, np.std])

        acc = gb['accuracy']

        y = acc['mean']
        _std = np.nan_to_num(acc['std'])
        y_lower = y - _std
        y_upper = y + _std

        x = np.arange(len(y))

        data[set_name]['y_lower'] = y_lower
        data[set_name]['y_upper'] = y_upper
        data[set_name]['x'] = x
        data[set_name]['y_mean'] = y
        data[set_name]['size'] = len(y)

    return data


def main():
    off.init_notebook_mode()
    output_name = 'accuracy.html'
    path_read = '/home/henry/Projects/eel/metadata'

    generate_df = pd.DataFrame([])
    select_df = pd.DataFrame([])
    integrate_df = pd.DataFrame([])

    files = os.listdir(path_read)
    for _file in files:
        if '_report_integrate' in _file:
            integrate_df = pd.read_csv(os.path.join(path_read, _file), sep=',')
        elif '_report_generate' in _file:
            generate_df = pd.read_csv(os.path.join(path_read, _file), sep=',')
        elif '_eda_select' in _file:
            select_df = pd.read_csv(os.path.join(path_read, _file), sep=',')

    step_names = filter(
        None,
        [bool(len(generate_df) > 0) * 'generate',
        bool(len(select_df) > 0) * 'select',
        bool(len(integrate_df) > 0) * 'integrate']
    )

    set_names = []
    for df in [generate_df, select_df, integrate_df]:
        try:
            set_names.extend(np.unique(df['set_name']))
        except KeyError:
            pass

    set_names = np.unique(set_names)

    colors = cm.viridis(np.linspace(0., 1., num=len(set_names) + 1))

    overall = {
        step_name: {
            set_name: dict(
                x=[],
                y_mean=[],
                y_lower=[],
                y_upper=[],
                color_std='rgba(?)'.replace('?', ','.join(map(str, list(colors[i][:-1]) + [0.2]))),
                color_mean='rgba(?)'.replace('?', ','.join(map(str, list(colors[i][:-1]) + [1.0]))),
                size=[]
            ) for i, set_name in enumerate(set_names)
        } for step_name in ['generate', 'select', 'integrate']
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
    shapes = []
    annotations = []

    _global_lower = np.inf
    _global_upper = -np.inf

    for key, data in overall.items():
        _lower_bound = min([
            min(data[set_name]['y_lower']) for set_name in set_names
        ])
        _upper_bound = max([
            max(data[set_name]['y_upper']) for set_name in set_names
        ])

        _global_lower = min(_lower_bound, _global_lower)
        _global_upper = max(_upper_bound, _global_upper)

    last_x = 0

    for step_name, bkg_color in it.izip(step_names, [256, 32, 256]):
        data = overall[step_name]
        
        for set_name in set_names:
            traces += [go.Scatter(
                x=last_x + data[set_name]['x'],
                y=data[set_name]['y_mean'],
                line=go.Line(color=data[set_name]['color_mean']),
                mode='lines',
                showlegend=True,
                name=step_name + ' ' + set_name,
            )]

            traces += [go.Scatter(
                x=list(last_x + data[set_name]['x']) + list(last_x + data[set_name]['x'])[::-1],  # appends
                y=list(data[set_name]['y_upper']) + list(data[set_name]['y_lower'])[::-1],  # appends
                fill='toself',
                fillcolor=data[set_name]['color_std'],
                line=go.Line(color=data[set_name]['color_std']),
                showlegend=False,
                hoverinfo='skip',
                name=None
            )]

        size = data[set_names[0]]['size'] - 1

        annotations += [
            dict(
                x=(last_x + (last_x + size)) / 2.,
                y=(_global_upper + _global_lower) / 2.,
                xref='x',
                yref='y',
                showarrow=False,
                text=step_name,
                # ax=0,
                # ay=-40
            )
        ]

        shapes += [
            {
                'type': 'rect',
                'y0': _global_lower,
                'y1': _global_upper,
                'x0': last_x,
                'x1': last_x + size,
                'line': {
                    'color': 'rgba(%f, %f, %f, 0.0)' % (bkg_color, bkg_color, bkg_color),
                },
                'fillcolor': 'rgba(%f, %f, %f, 0.1)' % (bkg_color, bkg_color, bkg_color),
            },
        ]
        last_x += size

    layout = dict(
        shapes=shapes,
        title='Accuracy in sets',
        xaxis=dict(
            title='Generation',
        ),
        yaxis=dict(
            title='Accuracy',
        ),
        annotations=annotations
    )

    fig = go.Figure(
        data=traces,
        layout=layout
    )
    off.plot(fig, filename=output_name)


if __name__ == '__main__':
    main()
