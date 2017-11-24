import itertools as it
import operator as op
import random

import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.colors import rgb2hex

from eda.core import get_fronts, a_dominates_b, get_flat_list, pairwise_domination

__author__ = 'Henry Cagnini'


def select(A, P, u, e):
    """

    :param A: old parent set
    :param P: candidate set
    :param u: minimum size
    :param e: approximation factor: the smaller it is, the small the tolerance for difference is.
    :return:
    """
    n_individuals, n_objectives = P.shape

    log2e = np.log2(e)

    A = np.array(map(tuple, A), dtype=[
        ('f0', np.float32),
        ('f1', np.float32)
    ])
    P = np.array(map(tuple, P), dtype=[
        ('f0', np.float32),
        ('f1', np.float32)
    ])

    for x in P:
        B = np.array([], dtype=[
            ('f0', np.float32),
            ('f1', np.float32)
        ])
        for y in A:
            equal = reduce(
                op.mul,
                [np.floor(np.log2(y[i]) / log2e) == np.floor(x[i] / log2e) for i in xrange(n_objectives)]
            )
            if equal:  # if solutions x and y are equal, given a tolerance interval
                B = np.hstack((B, y))

        if len(B) == 0:  # if there is no solution nearly equal to x
            A = np.hstack((A, x))  # add x to elite
        elif any([(a_dominates_b(y, x) == 1) for y in B]):  # if there is an solution in A that dominates x
            A = np.setdiff1d(A, np.hstack((B, x)))

    A_ = np.array([], dtype=[
        ('f0', np.float32),
        ('f1', np.float32)
    ])
    for y in A:
        add = True
        for z in A:
            if a_dominates_b(y, z) != 1:
                add = False
                break
        if add:
            A_ = np.hstack((A_, y))

    D = np.setdiff1d(A, A_)
    if len(A_) < u:
        to_add = u - len(A_)

        matrix = pairwise_domination(D, np.hstack((A_, D)))
        matrix = np.sort(matrix, axis=1)


        z = 0


        # TODO must add A_ based on their nondominance rank!

    return A_


def plot(fitness, fronts):
    n_fronts = len(fronts)
    print 'n_fronts:', n_fronts
    colors = cm.viridis(np.linspace(0, 1, len(fronts)))

    for i, front, color in it.izip(range(n_fronts), fronts, colors):
        inds = fitness[front]

        plt.scatter(inds[:, 0], inds[:, 1], c=rgb2hex(color), label='front %d' % i)

    plt.xlabel('$f_1(x)$')
    plt.ylabel('$f_2(x)$')
    plt.legend()
    plt.show()


def main():
    random.seed(0)
    np.random.seed(0)

    fitness = np.random.random((10, 2))
    fronts = get_fronts(fitness)

    A = fitness[fronts[0]]
    P = np.random.random((10, 2))
    u = 11  # TODO ???
    e = 0.5  # TODO ???

    select(A, P, u, e)


if __name__ == '__main__':
    main()
