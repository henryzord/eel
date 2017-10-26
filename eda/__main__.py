import json

from sklearn.metrics import accuracy_score

from eda.core import load_population
from eda.dataset import load_sets
from generation import generate
from selection import eda_select
from integration import integrate, __ensemble_predict__
from core import get_predictions

from sklearn.tree import DecisionTreeClassifier as clf


def main():
    params = json.load(open('../params.json', 'r'))

    X_train, X_val, X_test, y_train, y_val, y_test = load_sets(
        train_path=params['train_path'],
        val_path=params['val_path'],
        test_path=params['test_path'],
    )

    print 'generation:'

    ensemble, gen_pop, fitness = generate(
        X_train, y_train, X_val, y_val,
        base_classifier=clf,
        n_classifiers=params['n_individuals'],
        n_generations=params['n_generations']
    )

    val_predictions = get_predictions(ensemble, gen_pop, X_val)
    test_predictions = get_predictions(ensemble, gen_pop, X_test)

    print 'selection:'

    best_classifiers = eda_select(
        gen_pop, val_predictions, y_val,
        n_individuals=params['n_individuals'],
        n_generations=params['n_generations'],
    )

    print 'integration:'

    _best_weights = integrate(
        val_predictions[best_classifiers], y_val,
        n_individuals=params['n_individuals'], n_generations=params['n_generations'],
        test_predictions=test_predictions[best_classifiers],
        y_test=y_test
    )

    '''
        Now testing
    '''

    y_test_pred = __ensemble_predict__(_best_weights, test_predictions[best_classifiers])
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print 'test accuracy: %.2f' % test_accuracy



if __name__ == '__main__':
    main()
