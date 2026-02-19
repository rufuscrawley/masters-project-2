import os

from sklearn.model_selection import train_test_split

import optimisation

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from variables import names
import numpy as np
import pandas as pd
import os


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


norm_con_list = []


def normalise(self, column, log_norm=False, invert=False):
    if log_norm:
        self.loc[:, column] = self[column].map(lambda val: 0 if val <= 0 else np.log10(val))
    max_val = self[column].abs().max()
    self.loc[:, column] = self[column].map(lambda val: val / max_val)
    try:
        value = float(column)
        norm_con_list.append(max_val * value)
    except ValueError:
        norm_con_list.append(max_val)
    if invert:
        self.loc[:, column] = self[column].map(lambda val: val * -1)


def dropinf(self):
    self.replace([np.inf, -np.inf], np.nan, inplace=True)
    self.dropna(inplace=True)


def iloc_xy(self, data_split):
    x, y = self.iloc[:, :data_split], self.iloc[:, data_split:]
    return x, y


pd.DataFrame.normalise = normalise
pd.DataFrame.dropinf = dropinf
pd.DataFrame.iloc_xy = iloc_xy


def main(file_path: str, data_split: int, training_ratio: float,
         do_keras: bool = False, drop_values: list[str] = None):
    # Setup file structure
    create_directory("datasets/constants")
    create_directory("datasets/normalised")
    # Define file-path strings
    file = f"datasets/{file_path}.csv"
    n_file = f'datasets/normalised/n_{file_path}.csv'
    const_file = f'datasets/constants/const_{file_path}.csv'

    # Read in our csv file
    print("Reading in .csv...")
    csv = pd.read_csv(file)
    for value in drop_values:
        csv = csv.drop(value, axis=1)
    # Randomise rows
    csv.sample(frac=1)
    # Drop any inf values
    csv.dropinf()

    try:
        # If we already have a file to use, use this one
        n_csv = pd.read_csv(n_file)
        x_n, y_n = n_csv.iloc_xy(data_split)
        print("File found! Splitting data...")
    except FileNotFoundError:
        # Split it into I/O form
        x, y = csv.iloc_xy(data_split)

        # Normalise each input column
        print("Normalising input columns...")
        for key in names:
            if key in drop_values:
                continue
            x.normalise(key, names[key][0], names[key][1])
        # Normalise y-values
        print("Normalising output columns...")
        for col in y:
            y.normalise(col, log_norm=True, invert=True)

        print("Reconnecting csvs...")
        n_df = x.join(y)
        n_df.to_csv(n_file, index=False)

        print("Listing normalisation constants...")
        const_df = pd.DataFrame(norm_con_list)
        const_df.to_csv(const_file, index=False)
        print("Finishing up...")
        n_csv = pd.read_csv(n_file)
        x_n, y_n = n_csv.iloc_xy(data_split)

    # Create the validation set
    print("Creating validation set...")
    x_train, x_test, y_train, y_test = train_test_split(x_n, y_n, test_size=training_ratio, shuffle=False)
    training_length = len(x_train)
    split_point = int(np.floor(training_length * training_ratio))
    print(f"Splitting at {split_point}...")
    x_val, y_val = x_train[-split_point:], y_train[-split_point:]
    x_train, y_train = x_train[:-split_point], y_train[:-split_point]

    if do_keras:
        print("Running Keras!")
        optimisation.run_keras(x_train, y_train, x_val, y_val)
    else:
        print("Ignoring Keras!")
        optimisation.run_model(x_train, y_train, x_val, y_val, file_path)

    print("Creating test set!")
    test_set = x_test.join(y_test)
    test_set.to_csv(f'datasets/{file_path}_test.csv', index=False)
