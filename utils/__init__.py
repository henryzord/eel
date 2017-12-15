import arff
import numpy as np
import pandas as pd


def flatten(array):
    """
    Returns a flat list, made out of a list of lists.

    :param array: A list of lists.
    :return: A flat list.
    """
    return [item for sublist in array for item in sublist]


def path_to_arff(dataset_path):
    """
    Given a path to a dataset, reads and returns a dictionary which comprises an arff file.

    :type dataset_path: str
    :param dataset_path: Path to the dataset. Must contain the .arff file extension (i.e "my_dataset.arff")
    :rtype: dict
    :return: a dictionary with the arff dataset.
    """

    dataset_type = dataset_path.split('.')[-1].strip()
    assert dataset_type == 'arff', TypeError('Invalid type for dataset! Must be an \'arff\' file!')
    af = arff.load(open(dataset_path, 'r'))
    return af


def path_to_dataframe(path):
    """
    Given a path to an arff, transforms it to a pandas.DataFrame,
    whilst also inputting missing values with the mean for each column.

    :type path: str
    :param path:
    :return:
    :rtype: pandas.DataFrame
    """

    file_arff = path_to_arff(path)

    file_df = pd.DataFrame(
        data=file_arff['data'],
        columns=[x[0] for x in file_arff['attributes']],
    )

    file_df.replace('?', np.nan, inplace=True)

    for column in file_df.columns[:-1]:  # until last attribute
        file_df[column] = pd.to_numeric(file_df[column])
        file_df[column].fillna(file_df[column].mean(), inplace=True)

    return file_df


# def path_to_sets(path, train_size=0.5, val_size=0.25, test_size=0.25, random_state=None):
#     """
#
#     :param path:
#     :param train_size:
#     :param val_size:
#     :param test_size:
#     :param random_state:
#     :return: X_train, y_train, X_val, y_val, X_test, y_test
#     """
#
#     full_df = path_to_dataframe(path)
#
#     y_name = full_df.columns[-1]
#
#     full_df[y_name] = pd.Categorical(full_df[y_name])
#     full_df[y_name] = full_df[y_name].cat.codes
#
#     X = full_df[full_df.columns[:-1]]
#     y = full_df[full_df.columns[-1]]
#
#     if val_size <= 0:
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y,
#             train_size=0.75, test_size=0.25,
#             stratify=y, random_state=random_state
#         )
#         return X_train, y_train, X_test, y_test
#
#     # ------------------------------------------------------- #
#     # ------------------------------------------------------- #
#
#     X_train_val, X_test, y_train_val, y_test = train_test_split(
#         X, y,
#         train_size=(1. - test_size), test_size=test_size,
#         stratify=y, random_state=random_state
#     )
#
#     X_train, X_val, y_train, y_val = train_test_split(
#         X_train_val, y_train_val,
#         train_size=(train_size / (1. - test_size)), test_size=(1. - (train_size / (1. - test_size))),
#         stratify=y_train_val, random_state=random_state
#     )
#
#     return X_train, y_train, X_val, y_val, X_test, y_test
