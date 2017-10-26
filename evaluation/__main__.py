import json
from collections import namedtuple

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from eda.dataset import load_dataframe


def main():
    _dict = json.load(open('../params.json', 'r'))

    signature = ' '.join(_dict.keys())

    # Params = namedtuple('Params', signature)
    Params = namedtuple('Params', _dict.keys())
    params = Params(**_dict)

    full_df = load_dataframe(params.full_path)

    X = full_df[full_df.columns[:-1]]
    y = full_df[full_df.columns[-1]]

    acc_randomforest = []
    acc_adaboost = []

    n_all = X.shape[0]

    skf = StratifiedKFold(n_splits=params.n_folds, shuffle=True, random_state=params.random_state)
    for train_index, test_index in skf.split(X, y):
        X_train_val = X.iloc[train_index]
        X_test = X.iloc[test_index]

        y_train_val = y.iloc[train_index]
        y_test = y.iloc[test_index]

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            train_size=0.5, random_state=params.random_state
        )

        n_test = X_test.shape[0]

        adaboost = AdaBoostClassifier()
        randomforest = RandomForestClassifier()

        adaboost = adaboost.fit(X_train_val, y_train_val)  # type: AdaBoostClassifier
        randomforest = randomforest.fit(X_train_val, y_train_val)  # type: RandomForestClassifier

        preds_adaboost = adaboost.predict(X_test)
        preds_randomforest = randomforest.predict(X_test)

        acc_adaboost += [accuracy_score(y_test, preds_adaboost) * (float(n_test) / n_all)]
        acc_randomforest += [accuracy_score(y_test, preds_randomforest) * (float(n_test) / n_all)]

    print 'adaboost accuracy: %.2f' % sum(acc_adaboost)
    print 'randomForest accuracy: %.2f' % sum(acc_randomforest)


if __name__ == '__main__':
    main()

