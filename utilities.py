import os

import variables

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random
import numpy as np
import pandas as pd


def get_dataset_from_csv():
    random.seed()
    file = pd.read_csv(variables.file)
    n_file = pd.read_csv(variables.n_file)
    i, o = file.iloc[:, :15], file.iloc[:, 15:]
    n_i, n_o = n_file.iloc[:, :variables.split], n_file.iloc[:, variables.split:]

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


def get_x_consts(inputs):
    n_consts = np.array(pd.read_csv(variables.const_file).transpose())[0].tolist()
    n_consts = n_consts[:inputs]
    return n_consts


def get_y_consts(inputs):
    n_consts = np.array(pd.read_csv(variables.const_file).transpose())[0].tolist()
    n_consts = n_consts[inputs:]
    return n_consts
