import json

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier as clf

from core import check_distribution
from eda.core import load_population
from eda.dataset import load_sets


def select(ensemble, population, predictions, X_val, y_val):
    accs = np.sum(predictions, axis=1) / float(predictions.shape[1])

    median = np.median(accs)
    # # warnings.warn('warning: using whole population')
    to_select = np.ones(len(population), dtype=np.bool)
    # to_select = accs > median

    best_ensemble = ensemble[to_select]
    best_population = population[to_select]
    best_hit_or_miss = predictions[to_select]

    check_distribution(best_ensemble, best_population, X_val, y_val)

    return best_ensemble, best_population, best_hit_or_miss


if __name__ == '__main__':
    params = json.load(open('../params.json', 'r'))

    print 'loading datasets...'

    X_train, X_val, X_test, y_train, y_val, y_test = load_sets(
        params['train_path'],
        params['val_path'],
        params['test_path']
    )

    print 'loading population...'
    _population = pd.read_csv('generation_population.csv', sep=',').values

    _ensemble, _population, hit_or_miss = load_population(clf, _population, X_train, y_train, X_val, y_val)

    _best_classifiers, _best_features, _best_hit_or_miss = select(_ensemble, _population, hit_or_miss, X_val, y_val)

    _best_accs = np.sum(_best_hit_or_miss, axis=1) / float(_best_hit_or_miss.shape[1])

    print 'best base learners x features:', _best_features.shape
    print 'accuracies:'
    print 'min:\t %.7f' % np.min(_best_accs)
    print 'median:\t %.7f' % np.median(_best_accs)
    print 'mean:\t %.7f' % np.mean(_best_accs)
    print 'max:\t %.7f' % np.max(_best_accs)
