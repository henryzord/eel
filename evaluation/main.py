import json
import numpy as np
from sklearn.metrics import accuracy_score

from eda.dataset import load_sets
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier


def main():
    params = json.load(open('../params.json', 'r'))

    X_train, X_val, X_test, y_train, y_val, y_test = load_sets(
        params['train_path'],
        params['val_path'],
        params['test_path']
    )

    X_train_val = X_train.append(X_val, ignore_index=True)
    y_train_val = np.hstack((y_train, y_val))

    adaboost = AdaBoostClassifier()
    randomforest = RandomForestClassifier()

    adaboost = adaboost.fit(X_train_val, y_train_val)  # type: AdaBoostClassifier
    randomforest = randomforest.fit(X_train_val, y_train_val)  # type: RandomForestClassifier

    preds_adaboost = adaboost.predict(X_test)
    preds_randomforest = randomforest.predict(X_test)

    acc_adaboost = accuracy_score(y_test, preds_adaboost)
    acc_randomforest = accuracy_score(y_test, preds_randomforest)

    print 'Random forest test accuracy: %.2f' % acc_randomforest
    print 'AdaBoost test accuracy: %.2f' % acc_adaboost


if __name__ == '__main__':
    main()
