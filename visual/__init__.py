import plotly.graph_objs as go
import plotly.offline as off
import time
import numpy as np

output_name = 'out.html'

off.init_notebook_mode()

N = 500


random_x = np.linspace(0, 1, N)
random_y = np.random.randn(N)

time.sleep(5)

# Create a trace
trace = go.Scatter(
    x=random_x,
    y=random_y
)

data = [trace]

off.plot(
    dict(
        data=data,
        layout={
            'title': 'Test Plot',
            'font': dict(size=16)
        },
    ),
    filename=output_name,
    auto_open=False
)
