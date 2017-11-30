import itertools as it
import random

import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.colors import to_hex

from eda.core import get_fronts, select_operator


def main():
    random.seed(2)
    np.random.seed(2)

    # former elite begin
    A = np.random.random((10, 2))
    fronts = get_fronts(A)
    A = A[fronts[0]]
    # former elite end

    P = np.random.random((10, 2))

    Q = np.vstack((A, P))

    next_A = select_operator(
        len(A),
        len(P),
        Q,
        5,
        0.5
    )

    fronts = get_fronts(P)
    print fronts

    colors = cm.viridis(np.linspace(0, 1, num=len(fronts)))
    for i, front in enumerate(fronts):
        ind = P[front]
        plt.scatter(ind[:, 0], ind[:, 1], c=to_hex(colors[i]), s=45, label='front %d' % i)
        for j, (x, y) in it.izip(front, P[front]):
            plt.annotate(str(j), (x, y), (x+0.01, y+0.01))

    plt.scatter(Q[next_A, 0], Q[next_A, 1], s=80, facecolors='none', edgecolors='purple')

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
