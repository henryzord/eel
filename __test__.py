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

    P = np.random.random((10, 2))
    fronts = get_fronts(P)
    A = np.zeros(P.shape[0], dtype=np.bool)
    A[fronts[0]] = True

    next_A = select_operator(
        A,
        P,
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

    plt.scatter(P[next_A, 0], P[next_A, 1], s=80, facecolors='none', edgecolors='purple')

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
