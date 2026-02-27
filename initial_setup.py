import os

from sklearn.model_selection import train_test_split

import optimisation
import variables as v
from utilities import create_directory

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from variables import names
import numpy as np
import pandas as pd

n_consts = []


def normalise_parameters(self, key):
    inversion = -1 if v.names[key]["invert"] else 1
    if v.names[key]["logarithmic"]:
        self.loc[:, key] = self[key].map(lambda val: 0 if val <= 0 else np.log10(val))
    max_val = self[key].abs().max()
    self.loc[:, key] = self[key].map(lambda val: (val * inversion) / max_val)
    n_consts.append(max_val)


def normalise_fluxes(self, key):
    self.loc[:, key] = self[key].map(lambda val: 0 if val <= 0 else np.log10(val))
    max_val = self[key].abs().max()
    self.loc[:, key] = self[key].map(lambda val: (val * -1) / max_val)
    n_consts.append(max_val)


def dropinf(self):
    self.replace([np.inf, -np.inf], np.nan, inplace=True)
    self.dropna(inplace=True)


def iloc_xy(self, data_split):
    x, y = self.iloc[:, :data_split], self.iloc[:, data_split:]
    return x, y


pd.DataFrame.normalise_parameters = normalise_parameters
pd.DataFrame.normalise_fluxes = normalise_fluxes
pd.DataFrame.dropinf = dropinf
pd.DataFrame.iloc_xy = iloc_xy


def main(training_ratio: float, do_keras: bool = False):
    # Setup file structure
    create_directory("datasets/constants")
    create_directory("datasets/normalised")
    create_directory("models")

    # Read in our csv file
    print("Reading in .csv...")
    csv = pd.read_csv(v.file)
    for value in v.excluded:
        csv = csv.drop(value, axis=1)
    # Randomise rows
    csv.sample(frac=1)
    # Drop any inf values
    csv.dropinf()

    try:
        # If we already have a file to use, use this one
        n_csv = pd.read_csv(v.n_file)
        x_n, y_n = n_csv.iloc_xy(v.split)
        print("File found! Splitting data...")

    except FileNotFoundError:

        # Split it into I/O form
        x, y = csv.iloc_xy(v.split)
        # Normalise each input column
        print("Normalising inputs...")
        for key in names:
            if key in v.excluded:
                print(f"- Skipping {key}")
                continue
            print(f"Normalising {key}")
            x.normalise_parameters(key)
        # Normalise output fluxes
        print("Normalising output fluxes...")
        for col in y:
            y.loc[:, col] = y[col].map(lambda val: val / (float(col) * 10))
            y.normalise_fluxes(col)
        print("Listing normalisation constants...")
        const_df = pd.DataFrame(n_consts)
        const_df.to_csv(v.const_file, index=False)

        print("Finishing up...")
        n_csv = x.join(y)
        n_csv.to_csv(v.n_file, index=False)
        # No need to restitch / iloc, just use the old variables
        x_n, y_n = x, y

    # Create the validation set
    print("Creating validation set...")
    x_train, x_test, y_train, y_test = train_test_split(x_n, y_n,
                                                        test_size=training_ratio,
                                                        shuffle=True)
    split = int(np.floor(len(x_train) * training_ratio))
    x_val, y_val = x_train[-split:], y_train[-split:]
    x_train, y_train = x_train[:-split], y_train[:-split]

    if do_keras:
        print("Running Keras!")
        optimisation.run_keras(x_train, y_train, x_val, y_val)
    else:
        print("Ignoring Keras!")
        optimisation.run_model(x_train, y_train, x_val, y_val, v.filename)

    print("Creating test set!")
    test_set = x_test.join(y_test)
    test_set.to_csv(f'datasets/{v.filename}_test.csv', index=False)


main(0.2, False)
