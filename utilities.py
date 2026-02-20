import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random
import numpy as np
import pandas as pd


def get_dataset_from_csv(dataset_file, data_split):
    random.seed()
    file = pd.read_csv(f"datasets/{dataset_file}.csv")
    n_file = pd.read_csv(f"datasets/normalised/n_{dataset_file}.csv")
    i, o = file.iloc[:, :15], file.iloc[:, 15:]
    n_i, n_o = n_file.iloc[:, :data_split], n_file.iloc[:, data_split:]

    ROW = random.randint(1, 50_000)
    return (np.array(i.iloc[ROW]), np.array(o.iloc[ROW]),
            np.array(n_i.iloc[ROW]), np.array(n_o.iloc[ROW]))


def create_directory(folder_name):
    """
    Tries to write a folder. If the folder already exists, suppresses the error, and continues.
    :type folder_name: str
    :param folder_name: The folder name.
    """
    try:
        os.makedirs(folder_name)
    except OSError:
        pass
