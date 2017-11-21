import json

import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from eda import eel, Reporter
from eda.dataset import path_to_dataframe
from datetime import datetime as dt

import numpy as np


def main():
    params = json.load(open('../params.json', 'r'))

    full_df = path_to_dataframe(params['full_path'])

    y_name = full_df.columns[-1]

    full_df[y_name] = pd.Categorical(full_df[y_name])
    full_df[y_name] = full_df[y_name].cat.codes

    X = full_df[full_df.columns[:-1]]
    y = full_df[full_df.columns[-1]]

    acc_randomforest = []
    acc_adaboost = []
    acc_eel = []

    n_all = X.shape[0]

    skf = StratifiedKFold(n_splits=params['n_folds'], shuffle=True, random_state=params['random_state'])

    date = dt.now()

    for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
        _run_eel = []
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

            preds_eel = eel(params['metaparams'], X_train, y_train, X_val, y_val, X_test, y_test, reporter=reporter)
            _run_eel += [accuracy_score(y_test, preds_eel) * (float(n_test) / n_all)]

        acc_eel += [np.mean(_run_eel)]

        adaboost = AdaBoostClassifier()
        randomforest = RandomForestClassifier()

        adaboost = adaboost.fit(X_train_val, y_train_val)  # type: AdaBoostClassifier
        randomforest = randomforest.fit(X_train_val, y_train_val)  # type: RandomForestClassifier

        preds_adaboost = adaboost.predict(X_test)
        preds_randomforest = randomforest.predict(X_test)

        acc_adaboost += [accuracy_score(y_test, preds_adaboost) * (float(n_test) / n_all)]
        acc_randomforest += [accuracy_score(y_test, preds_randomforest) * (float(n_test) / n_all)]

        print '----------------------------------'
        print 'partition accuracies:'
        print '----------------------------------'
        print '\tadaboost accuracy: %.4f' % acc_adaboost[-1]
        print '\trandomForest accuracy: %.4f' % acc_randomforest[-1]
        print '\teel accuracy: %.4f' % acc_eel[-1]
        print '----------------------------------'
        print '----------------------------------'

    print 'adaboost accuracy: %.4f' % sum(acc_adaboost)
    print 'randomForest accuracy: %.4f' % sum(acc_randomforest)
    print 'eel accuracy: %.4f' % sum(acc_eel)


if __name__ == '__main__':
    main()
