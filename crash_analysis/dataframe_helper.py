import os
import six

import pandas as pd


def remove_empty(df):
    """Remove rows with any empty (NaN/None) values"""
    return df.dropna(how='any')


def fill_empty(df):
    """Fill rows with empty (NaN/None) values with ' '"""
    return df.fillna(value=' ')


def is_df_set(df):
    """
    Test to see if variable is None or has a dataframe object
    :param df: dataframe to test
    :return: boolean
    """
    # type(df) is not type(None)  # old version
    return isinstance(df, type(None))


def get_columns(df):
    """
    Returns list of column titles for input or object dataframe.
    :param df: dataframe
    :return: list of strings -- titles of dataframe columns
    """
    assert isinstance(df, pd.DataFrame)
    return [c for c in df.columns]


def read_csv(csv_file_path):
    """
    Reads a CSV file, converts to a pandas dataframe object.
    :param csv_file_path: text path to csv file to read
    :return: None
    :side_effects: Stores local dataframe to object
    """

    pwd = os.getcwd()
    os.chdir(os.path.dirname(csv_file_path))

    df = pd.read_csv(os.path.basename(csv_file_path), low_memory=False)

    os.chdir(pwd)
    return df


def get_column(df, col_name):
    """ Get column of a dataframe"""
    assert isinstance(df, pd.DataFrame)
    assert isinstance(col_name, six.string_types)

    return df[col_name]


def filter_dataframe(df, **kwargs):
    """
    Generalized method to select the desired row(s) and column(s) from the dataframe. TO BE IMPLEMENTED
    :param df:
    :param kwargs:
    :return:
    """
    working_df = df.copy()

    for key, value in kwargs.items():
        value = value if isinstance(value, list) else [value]
        working_df = working_df[working_df[str(key)].isin(value)]

    return working_df