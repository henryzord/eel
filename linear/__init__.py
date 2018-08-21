from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np

__author__ = 'Henry Cagnini'


class LogisticAdaBoost(AdaBoostClassifier):
    """
    This class implements AdaBoost with a logistic regression for each class, in order to fine tune the
    voting weights of AdaBoost. For binary datasets, we use only one logistic regression.
    """

    def fit(self, X, y, sample_weight=None):
        """
        Builds a ensemble with classifiers from AdaBoost and voting weights optimized by logistic regression.

        :type X: ({array-like, sparse matrix} of shape = [n_samples, n_features])
        :param X: The training input samples. Sparse matrix can be CSC, CSR, COO, DOK, or LIL.
            DOK and LIL are converted to CSR.
        :type y: (array-like of shape = [n_samples])
        :param y: The target values (class labels).
        :type sample_weight: (array-like of shape = [n_samples], optional)
        :param sample_weight: Sample weights. If None, the sample weights are initialized to 1 / n_samples.
        :return: Returns self.
        """

        super(LogisticAdaBoost, self).__init__(algorithm='SAMME')
        super(LogisticAdaBoost, self).fit(X, y, sample_weight=sample_weight)

        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        self.logistic_model = []
        all_preds = self.get_predictions(X)

        for that_class in self.classes:
            if self.n_classes == 2:
                self.logistic_model += [LogisticRegression().fit(all_preds.T, y)]
                break
            else:
                binary_preds = (all_preds == that_class).astype(np.int32)
                self.logistic_model += [LogisticRegression().fit(binary_preds.T, y == that_class)]

        return self

    def predict(self, X):
        all_preds = self.get_predictions(X)

        global_votes = np.empty((len(X), self.n_classes), dtype=np.float32)

        for i, that_class in enumerate(self.classes):
            if self.n_classes > 2:
                classes_ = self.logistic_model[i].classes_
                right_index = np.argmax(classes_)
                binary_preds = (all_preds == that_class).astype(np.int32)
                global_votes[:, i] = self.logistic_model[i].predict_proba(binary_preds.T)[:, right_index]
            else:
                classes_ = np.int32(self.logistic_model[0].classes_)
                proba = self.logistic_model[0].predict_proba(all_preds.T)
                global_votes[:, classes_] = proba[:, classes_]
                break

        final_preds = np.argmax(global_votes, axis=1)
        return final_preds

    def get_predictions(self, X):
        """
        Given a list of classifiers and the features each one of them uses,
        returns a matrix of predictions for dataset X.

        :param X: A dataset comprised of instances and attributes.
        :param preds: optional - matrix where each row is a classifier and each column an instance.
        :return: An matrix where each row is a classifier and each column an instance in X.
        """

        preds = np.empty((self.n_estimators, X.shape[0]), dtype=np.int32)

        for i in xrange(self.n_estimators):  # number of base classifiers
            preds[i, :] = self.estimators_[i].predict(X)

        return preds


class AdaBoostOnes(AdaBoostClassifier):
    """
    This class implements AdaBoost with a logistic regression for each class, in order to fine tune the
    voting weights of AdaBoost. For binary datasets, we use only one logistic regression.
    """

    def fit(self, X, y, sample_weight=None):
        """
        Builds a ensemble with classifiers from AdaBoost and voting weights optimized by logistic regression.

        :type X: ({array-like, sparse matrix} of shape = [n_samples, n_features])
        :param X: The training input samples. Sparse matrix can be CSC, CSR, COO, DOK, or LIL.
            DOK and LIL are converted to CSR.
        :type y: (array-like of shape = [n_samples])
        :param y: The target values (class labels).
        :type sample_weight: (array-like of shape = [n_samples], optional)
        :param sample_weight: Sample weights. If None, the sample weights are initialized to 1 / n_samples.
        :return: Returns self.
        """

        super(AdaBoostOnes, self).__init__(algorithm='SAMME')
        super(AdaBoostOnes, self).fit(X, y, sample_weight=sample_weight)
        self.estimator_weights_ = np.ones(len(self.estimator_weights_))
        return self

    def get_predictions(self, X):
        """
        Given a list of classifiers and the features each one of them uses,
        returns a matrix of predictions for dataset X.

        :param X: A dataset comprised of instances and attributes.
        :param preds: optional - matrix where each row is a classifier and each column an instance.
        :return: An matrix where each row is a classifier and each column an instance in X.
        """

        preds = np.empty((self.n_estimators, X.shape[0]), dtype=np.int32)

        for i in xrange(self.n_estimators):  # number of base classifiers
            preds[i, :] = self.estimators_[i].predict(X)

        return preds

class AdaBoostNormal(AdaBoostClassifier):
    """
    This class implements AdaBoost with a logistic regression for each class, in order to fine tune the
    voting weights of AdaBoost. For binary datasets, we use only one logistic regression.
    """

    def fit(self, X, y, sample_weight=None):
        """
        Builds a ensemble with classifiers from AdaBoost and voting weights optimized by logistic regression.

        :type X: ({array-like, sparse matrix} of shape = [n_samples, n_features])
        :param X: The training input samples. Sparse matrix can be CSC, CSR, COO, DOK, or LIL.
            DOK and LIL are converted to CSR.
        :type y: (array-like of shape = [n_samples])
        :param y: The target values (class labels).
        :type sample_weight: (array-like of shape = [n_samples], optional)
        :param sample_weight: Sample weights. If None, the sample weights are initialized to 1 / n_samples.
        :return: Returns self.
        """

        super(AdaBoostNormal, self).__init__(algorithm='SAMME')
        super(AdaBoostNormal, self).fit(X, y, sample_weight=sample_weight)
        self.estimator_weights_ = np.random.normal(1,0.25,len(self.estimator_weights_))
        return self

    def get_predictions(self, X):
        """
        Given a list of classifiers and the features each one of them uses,
        returns a matrix of predictions for dataset X.

        :param X: A dataset comprised of instances and attributes.
        :param preds: optional - matrix where each row is a classifier and each column an instance.
        :return: An matrix where each row is a classifier and each column an instance in X.
        """

        preds = np.empty((self.n_estimators, X.shape[0]), dtype=np.int32)

        for i in xrange(self.n_estimators):  # number of base classifiers
            preds[i, :] = self.estimators_[i].predict(X)

        return preds