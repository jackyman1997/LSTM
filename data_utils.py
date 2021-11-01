from os import listdir
from pandas import DataFrame, read_csv
from pandas.core.series import Series
from numpy import ndarray
import typing


def get_csv_data(folder_path: str) -> typing.List[DataFrame]:
    data = []
    files = listdir(folder_path)
    for file in files:
        if file.endswith('.csv'):
            df = read_csv(folder_path+file)
            data.append(df)
    return data


def get_data_local():
    return NotImplementedError


def get_data_S3():
    return NotImplementedError


def get_data_RDS():
    return NotImplementedError


def diff_seq(
    seq: typing.Union[list, ndarray, Series],
    interval: int = 1,
    same_length: bool = False
) -> Series:
    output = [seq[i] - seq[i - interval] for i in range(interval, len(seq))]
    if same_length:
        for i in range(interval):
            output.insert(0, 0)
    return Series(output)
