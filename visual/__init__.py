import plotly.graph_objs as go
import plotly.offline as off
import time
import numpy as np
import pandas as pd
import os

from matplotlib import cm


def __plot__(data, df):
    set_names = df['set_name'].unique()
    colors = cm.viridis(np.linspace(0., 1., num=len(set_names)))
    for i, set_name in enumerate(set_names):
        _set = df.loc[df['set_name'] == set_name]

        gb = _set.groupby(by=['method', 'fold', 'generation', 'set_name', 'set_size'])  # type: pd.core.groupby.GroupBy

        acc = gb['accuracy'].agg([np.mean, np.std])

        y = acc['mean']
        _std = np.nan_to_num(acc['std'])
        y_lower = list(acc['mean'] - _std)
        y_upper = list(acc['mean'] + _std)
        x = range(len(y))

        # Create a trace
        colors[i][-1] = 0.2
        rgba_std = 'rgba(?)'.replace('?', ','.join(map(str, colors[i])))
        colors[i][-1] = 1.
        rgba_mean = 'rgba(?)'.replace('?', ','.join(map(str, colors[i])))

        # TODO here!
        data[set_name] += [trace_std, trace_mean]
    return data


def main():

    off.init_notebook_mode()
    output_name = 'out.html'
    path_read = '/home/henry/Projects/eel/metadata'

    # TODO must support absence of some reports, or even incompletude!

    files = os.listdir(path_read)
    for _file in files:
        if '_report_integrate' in _file:
            integrate_df = pd.read_csv(os.path.join(path_read, _file), sep=',')
        elif '_report_generate' in _file:
            generate_df = pd.read_csv(os.path.join(path_read, _file), sep=',')
        elif '_eda_select' in _file:
            select_df = pd.read_csv(os.path.join(path_read, _file), sep=',')

    data = dict(
        x=[],
        y_mean=[],
        y_lower=[],
        y_upper=[]
    )
    # TODO plot!
    # data = __plot__(data, generate_df)
    # data = __plot__(data, select_df)
    data = __plot__(data, integrate_df)

    trace_mean = go.Scatter(
        x=x,  # appends
        y=y,  # appends
        line=go.Line(color=rgba_mean),
        mode='lines',
        showlegend=True,
        name=set_name,
    )

    trace_std = go.Scatter(
        x=x + x[::-1],  # appends
        y=y_upper + y_lower[::-1],  # appends
        fill='tozerox',
        fillcolor=rgba_std,
        line=go.Line(color=rgba_std),
        showlegend=False,
    )

    layout = go.Layout(
        paper_bgcolor='rgb(255,255,255)',
        plot_bgcolor='rgb(229,229,229)',
        xaxis=go.XAxis(
            gridcolor='rgb(255,255,255)',
            range=[1, 10],
            showgrid=True,
            showline=False,
            showticklabels=True,
            tickcolor='rgb(127,127,127)',
            ticks='outside',
            zeroline=False
        ),
        yaxis=go.YAxis(
            gridcolor='rgb(255,255,255)',
            showgrid=True,
            showline=False,
            showticklabels=True,
            tickcolor='rgb(127,127,127)',
            ticks='outside',
            zeroline=False
        ),
    )
    fig = go.Figure(data=data, layout=layout)
    off.plot(fig, filename=output_name)


if __name__ == '__main__':
    main()

# off.plot(
#     dict(
#         data=data,
#         layout={
#             'title': 'Accuracy on sets throughout evolution',
#             'font': dict(size=16)
#         },
#     ),
#     filename=output_name,
#     auto_open=True
# )


