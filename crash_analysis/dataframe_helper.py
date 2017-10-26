import os

import pandas as pd
from typing import List, Any


def remove_empty(df: pd.DataFrame):
    """Remove rows with any empty (NaN/None) values
    
    >>> df = pd.DataFrame({'A': list(range(0, 10)), 'B': list(range(10,20))})
    >>> df['A'][1] = None
    >>> df
         A   B
    0  0.0  10
    1  NaN  11
    2  2.0  12
    3  3.0  13
    4  4.0  14
    5  5.0  15
    6  6.0  16
    7  7.0  17
    8  8.0  18
    9  9.0  19
    >>> remove_empty(df)
         A   B
    0  0.0  10
    2  2.0  12
    3  3.0  13
    4  4.0  14
    5  5.0  15
    6  6.0  16
    7  7.0  17
    8  8.0  18
    9  9.0  19
    """
    return df.dropna(how='any')


def fill_empty(df: pd.DataFrame, val: Any = ' ') -> pd.DataFrame:
    """Fill rows with empty (NaN/None) values with ' '
    
    >>> df = pd.DataFrame({'A': list(range(0, 10)), 'B': list(range(10,20))})
    >>> df['A'][1] = None
    >>> df
         A   B
    0  0.0  10
    1  NaN  11
    2  2.0  12
    3  3.0  13
    4  4.0  14
    5  5.0  15
    6  6.0  16
    7  7.0  17
    8  8.0  18
    9  9.0  19
    >>> fill_empty(df, 0)
         A   B
    0  0.0  10
    1  0.0  11
    2  2.0  12
    3  3.0  13
    4  4.0  14
    5  5.0  15
    6  6.0  16
    7  7.0  17
    8  8.0  18
    9  9.0  19
    """
    return df.fillna(value=val)


def is_df_set(df: pd.DataFrame) -> bool:
    """Test to see if variable is None or has a dataframe object
    
    :param df: dataframe to test
    :return: boolean
    
    >>> df = pd.DataFrame({'A': list(range(0, 10)), 'B': list(range(10,20))})
    >>> is_df_set(df)
    True
    >>> is_df_set(None)
    False
    """
    # type(df) is not type(None)  # old version
    return not isinstance(df, type(None))


def get_columns(df: pd.DataFrame) -> List[str]:
    """ Returns list of column titles for input or object dataframe.
    
    :param df: dataframe
    :return: list of strings -- titles of dataframe columns
    
    >>> get_columns(list(range(10)))
    Traceback (most recent call last):
    ...
    AssertionError
    >>> df = pd.DataFrame({'A': list(range(0, 10)), 'B': list(range(10,20))})
    >>> get_columns(df)
    ['A', 'B']
    """
    assert isinstance(df, pd.DataFrame)
    return [c for c in df.columns]


def read_csv(csv_file_path: str) -> pd.DataFrame:
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


def get_column(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """Get column of a dataframe
    
    
    :param df: source dataframe
    :param col_name: column in dataframe
    :return: dataframe of N rows and 1 column
    
    >>> df = pd.DataFrame({'A': list(range(0, 10)), 'B': list(range(10,20))})
    >>> df
       A   B
    0  0  10
    1  1  11
    2  2  12
    3  3  13
    4  4  14
    5  5  15
    6  6  16
    7  7  17
    8  8  18
    9  9  19
    >>> get_column(df, 'A')
    0    0
    1    1
    2    2
    3    3
    4    4
    5    5
    6    6
    7    7
    8    8
    9    9
    Name: A, dtype: int64
    >>> get_column(df, 'B')
    0    10
    1    11
    2    12
    3    13
    4    14
    5    15
    6    16
    7    17
    8    18
    9    19
    Name: B, dtype: int64
    >>> get_column(list(range(10)), 'A')
    Traceback (most recent call last):
    ...
    AssertionError
    >>> get_column(df, 'C')
    Traceback (most recent call last):
    ...
    AssertionError
    """
    assert isinstance(df, pd.DataFrame)
    assert col_name in df.columns

    return df[col_name]


def filter_dataframe(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Filters dataframe rows by key-value pairs.
    
    Generalized method to select the desired row(s) and column(s) from the dataframe. TO BE IMPLEMENTED
    :param df: source dataframe
    :param kwargs: key-value pairs that represent. Values can be a scalar or list of values. 
    :return: a filtered copy of the data. 
    
    >>> df = pd.DataFrame({'A': list(range(0, 10)), 'B': list(range(10,20))})
    >>> df
       A   B
    0  0  10
    1  1  11
    2  2  12
    3  3  13
    4  4  14
    5  5  15
    6  6  16
    7  7  17
    8  8  18
    9  9  19
    >>> filter_dataframe(df, A=[1, 2, 4, 6, 8])
       A   B
    1  1  11
    2  2  12
    4  4  14
    6  6  16
    8  8  18
    >>> filter_dataframe(df, B=15)
       A   B
    5  5  15
    """
    working_df = df.copy()

    for key, value in kwargs.items():
        value = value if isinstance(value, list) else [value]
        working_df = working_df[working_df[str(key)].isin(value)]

    return working_df

if __name__ == '__main__':
    import doctest

    doctest.testmod()
