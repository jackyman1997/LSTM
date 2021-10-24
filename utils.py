from os import listdir
from pandas import DataFrame, read_csv


def get_csv_data(folder_path: str) -> DataFrame:
    data = []
    files_paths = listdir(folder_path)
    for filename in files_paths:
        filepath = folder_path + "/" + filename
        data.append(read_csv(filepath))
    return data
