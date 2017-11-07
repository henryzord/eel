import arff
import pandas as pd
import numpy as np
import itertools as it

from sklearn.model_selection import train_test_split


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


def path_to_sets(path, train_size=0.5, val_size=0.25, test_size=0.25, random_state=None):
    full_df = path_to_dataframe(path)

    y_name = full_df.columns[-1]

    full_df[y_name] = pd.Categorical(full_df[y_name])
    full_df[y_name] = full_df[y_name].cat.codes

    X = full_df[full_df.columns[:-1]]
    y = full_df[full_df.columns[-1]]

    if val_size <= 0:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            train_size=0.75, test_size=0.25,
            stratify=y, random_state=random_state
        )
        return X_train, y_train, X_test, y_test

    # ------------------------------------------------------- #
    # ------------------------------------------------------- #

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        train_size=(1. - test_size), test_size=test_size,
        stratify=y, random_state=random_state
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        train_size=(train_size / (1. - test_size)), test_size=(1. - (train_size / (1. - test_size))),
        stratify=y_train_val, random_state=random_state
    )

    return X_train, y_train, X_val, y_val, X_test, y_test

# def make_and_write_sets(path, sizes):
#     df = load_dataframe(path)
#
#     print 'full size:', df.shape
#
#     classes = np.unique(df[df.columns[-1]])
#
#     dict_converter = {x: i for i, x in enumerate(classes)}
#
#     df[df.columns[-1]] = map(lambda x: dict_converter[x], df[df.columns[-1]])
#
#     indices = range(df.shape[0])
#     np.random.shuffle(indices)
#
#     for name, size in sizes.items():
#         set_indices = np.random.choice(indices, size=int(len(indices) * size), replace=False)
#
#         indices = list(set(indices) - set(set_indices))
#
#         _set = df.iloc[set_indices]
#         _set.to_csv(name + '.csv', index=False, sep=',')


# def load_sets(train_path, val_path, test_path):
#     train_df = load_dataframe(train_path)
#     val_df = load_dataframe(val_path)
#     test_df = load_dataframe(test_path)
#
#     X_train = train_df[train_df.columns[:-1]]
#     y_train = train_df[train_df.columns[-1]]
#
#     X_val = val_df[val_df.columns[:-1]]
#     y_val = val_df[val_df.columns[-1]]
#
#     X_test = test_df[test_df.columns[:-1]]
#     y_test = test_df[test_df.columns[-1]]
#
#     classes = y_train.unique()
#
#     dict_converter = {x: i for i, x in enumerate(classes)}
#
#     y_train = np.array([dict_converter[j] for j in y_train])
#     y_val = np.array([dict_converter[j] for j in y_val])
#     y_test = np.array([dict_converter[j] for j in y_test])
#
#     return X_train, X_val, X_test, y_train, y_val, y_test

