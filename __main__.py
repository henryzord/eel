import json
import numpy as np
import pandas as pd
from datetime import datetime as dt

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier as clf

from eda import Reporter, Ensemble
from utils import path_to_dataframe
from eda.generation import EnsembleGenerator
from eda.selection import select


def eelem(params, X_train, y_train, X_val, y_val, X_test, y_test, reporter=None):
    print '-------------------------------------------------------'
    print '--------------------- generation ----------------------'
    print '-------------------------------------------------------'

    gen_inst = EnsembleGenerator(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        base_classifier=clf
    )

    classifiers, features, fitness = gen_inst.generate(
        n_classifiers=params['generation']['n_individuals'],
        n_generations=params['generation']['n_generations'],
        selection_strength=params['generation']['selection_strength'],
        reporter=reporter
    )

    val_predictions = get_predictions(classifiers, features, X_val)
    test_predictions = get_predictions(classifiers, features, X_test)

    selected_classifiers = select(
        features=features, classifiers=classifiers,
        val_predictions=val_predictions, y_val=y_val,
        n_individuals=params['selection']['n_individuals'],
        n_generations=params['selection']['n_generations'],
        reporter=reporter
    )

    raise NotImplementedError('not implemented yet!')

    best_classifiers = np.ones(len(classifiers), dtype=np.bool)
    _best_weights = np.ones((len(best_classifiers), len(np.unique(y_val))), dtype=np.float32)

    '''
        Now testing
    '''

    y_test_pred = get_classes(_best_weights, test_predictions[np.where(best_classifiers)])
    return y_test_pred


def main():
    params = json.load(open('params.json', 'r'))

    full_df = path_to_dataframe(params['full_path'])

    y_name = full_df.columns[-1]

    full_df[y_name] = pd.Categorical(full_df[y_name])
    full_df[y_name] = full_df[y_name].cat.codes

    X = full_df[full_df.columns[:-1]]
    y = full_df[full_df.columns[-1]]

    acc_randomforest = []
    acc_adaboost = []
    acc_eelem = []

    std_randomforest = []
    std_adaboost = []
    std_eelem = []

    n_all = X.shape[0]

    skf = StratifiedKFold(n_splits=params['n_folds'], shuffle=True, random_state=params['random_state'])

    date = dt.now()

    for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
        _run_eelem = []
        _run_adaboost = []
        _run_randomforest = []
        for run in xrange(params['n_runs']):

            X_train_val = X.iloc[train_index]
            X_test = X.iloc[test_index]

            y_train_val = y.iloc[train_index]
            y_test = y.iloc[test_index]

            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val,
                train_size=0.5, test_size=0.5, random_state=params['random_state']
            )

            reporter = Reporter(
                Xs=[X_train, X_val, X_test],
                ys=[y_train, y_val, y_test],
                set_names=['train', 'val', 'test'],
                output_path=params['reporter_output'],
                date=date,
                fold=fold,
                n_run=run
            )

            n_test = X_test.shape[0]

            adaboost = AdaBoostClassifier()
            randomforest = RandomForestClassifier()

            adaboost = adaboost.fit(X_train_val, y_train_val)  # type: AdaBoostClassifier
            randomforest = randomforest.fit(X_train_val, y_train_val)  # type: RandomForestClassifier

            preds_eel = eelem(params['metaparams'], X_train, y_train, X_val, y_val, X_test, y_test, reporter=reporter)
            preds_adaboost = adaboost.predict(X_test)
            preds_randomforest = randomforest.predict(X_test)

            __acc_eelem = accuracy_score(y_test, preds_eel)
            __acc_adaboost = accuracy_score(y_test, preds_adaboost)
            __acc_randomforest = accuracy_score(y_test, preds_randomforest)

            # -------- accuracy for that run -------- #
            _run_eelem += [__acc_eelem * (float(n_test) / n_all)]
            _run_adaboost += [__acc_adaboost * (float(n_test) / n_all)]
            _run_randomforest += [__acc_randomforest * (float(n_test) / n_all)]

        # -------- accuracy for that fold -------- #
        acc_eelem += [np.mean(_run_eelem)]  # the accuracy for eelem in that fold is the mean for N runs
        acc_adaboost += [np.mean(_run_adaboost)]
        acc_randomforest += [np.mean(_run_randomforest)]

        std_eelem += [np.std(_run_eelem)]
        std_adaboost += [np.std(_run_adaboost)]
        std_randomforest += [np.std(_run_randomforest)]

        print '----------------------------------'
        print '------ partition accuracies: -----'
        print '----------------------------------'
        print '\tadaboost accuracy: %.4f +- %.4f' % (acc_adaboost[-1], std_adaboost[-1])
        print '\trandomForest accuracy: %.4f +- %.4f' % (acc_randomforest[-1], std_randomforest[-1])
        print '\teelem accuracy: %.4f +- %.4f' % (acc_eelem[-1], std_eelem[-1])
        print '----------------------------------'
        print '----------------------------------'

    print 'adaboost accuracy: %.4f +- %.4f' % (sum(acc_adaboost), np.mean(std_adaboost))
    print 'randomForest accuracy: %.4f +- %.4f' % (sum(acc_randomforest), np.mean(std_randomforest))
    print 'eelem accuracy: %.4f +- %.4f' % (sum(acc_eelem), np.mean(std_eelem))


if __name__ == '__main__':
    main()
