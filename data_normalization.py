"""
You must override the class in this script for defining a data normalization strategy.
"""


__author__ = 'Henry Cagnini'

from sklearn.preprocessing import MinMaxScaler


class DataNormalizer(MinMaxScaler):
    def __init__(self, feature_range=(0, 1), copy=True):
        super().__init__(feature_range, copy)

    def fit(self, X, y=None):
        return super().fit(X, y)

    def transform(self, X):
        return super().transform(X)

    def fit_transform(self, X, y=None, **fit_params):
        return super().fit_transform(X, y, **fit_params)

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #
    # def fit(self, X, y=None):
    #     super().fit(X, y)
    #
    # def fit_transform(self, X, y=None, **fit_params):
    #     super().fit_transform(X, y, **fit_params)
    #
    # def transform(self, X):
    #     super().transform(X)
