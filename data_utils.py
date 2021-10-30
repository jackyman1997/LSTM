from os import listdir
from pandas import DataFrame, read_csv
import typing


def get_csv_data(folder_path: str) -> typing.List[DataFrame]:
    data = []
    files_paths = listdir(folder_path)
    for filename in files_paths:
        filepath = folder_path + "/" + filename
        data.append(read_csv(filepath))
    return data


def get_data_local():
    return NotImplementedError


def get_data_S3():
    return NotImplementedError


def get_data_RDS():
    return NotImplementedError
