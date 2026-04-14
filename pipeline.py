import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import networks
import variables as v
from utilities import create_directory

norm_con_list = []


def normalise(self, col, log=False):
    if log:
        self.loc[:, col] = np.log10(self[col])
    max_val = max(self[col], key=abs)
    self.loc[:, col] = self[col].map(lambda val: val / max_val)
    norm_con_list.append(max_val)


def dropinf(self):
    self.replace([np.inf, -np.inf], np.nan, inplace=True)
    self.replace(0, np.nan, inplace=True)
    self.dropna(inplace=True)


def iloc_xy(self, data_split):
    x, y = self.iloc[:, :data_split], self.iloc[:, data_split:]
    return x, y


pd.DataFrame.normalise = normalise
pd.DataFrame.dropinf = dropinf
pd.DataFrame.iloc_xy = iloc_xy


def initiate(training_ratio: float, do_keras: bool = False):
    # Setup file structure
    create_directory("datasets/normalised")
    create_directory("models")

    # Read in our csv file
    print("Reading in .csv...")
    csv = pd.read_csv(v.file)
    for value in v.names.keys():
        if value not in v.included:
            csv = csv.drop(value, axis=1)
    # Drop any inf values
    csv.dropinf()

    try:
        # If we already have a file to use, use this one
        n_csv = pd.read_csv(v.n_file)
        x_n, y_n = n_csv.iloc_xy(v.split)
        print("File found! Splitting data...")
    except FileNotFoundError:
        # Split it into I/O form
        print(f"Splitting at {v.split}")
        x, y = csv.iloc_xy(v.split)
        # Normalise each input column
        for col in x:
            if col in v.included:
                x.normalise(col, v.names[col])
        print("Normalising output fluxes...")
        for col in y:
            y.loc[:, col] = y[col].map(lambda val: val / (float(col) * 10))
            y.normalise(col, True)

        print("Listing normalisation constants...")
        const_df = pd.DataFrame(norm_con_list)
        const_df.to_csv(v.const_file, index=False)
        print("Finishing up...")
        n_csv = x.join(y)
        n_csv.to_csv(v.n_file, index=False)
        # No need to restitch / iloc, just use the old variables
        x_n, y_n = x, y

    # Create the validation set
    print("Creating validation set...")
    (x_train, x_test,
     y_train, y_test) = train_test_split(x_n, y_n, test_size=training_ratio)

    if do_keras:
        print("Running Keras!")
        # optimisation.run_keras(x_train, y_train)
    else:
        print("Ignoring Keras!")
        networks.run_model(x_train, y_train)
    print("Creating test set!")
    test_set = x_test.join(y_test)
    test_set.to_csv(f'datasets/{v.filename}_test.csv', index=False)
