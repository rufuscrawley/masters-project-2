import os

import keras_tuner
from matplotlib import pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import keras
import variables_early as v

norm_con_list = []


def normalise(self, col, log=False):
    if log:
        self.loc[:, col] = np.log10(self[col])
    mean = self[col].mean()
    std = self[col].std()
    self.loc[:, col] = self[col].map(lambda val: (val - mean) / std)
    norm_con_list.append((mean, std))


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
        run_keras(x_train, y_train)
    else:
        print("Ignoring Keras!")
        run_model(x_train, y_train)
    print("Creating test set!")
    test_set = x_test.join(y_test)
    test_set.to_csv(f'datasets/{v.filename}_test.csv', index=False)


def run_model(x_train, y_train) -> None:
    model: keras.Model = keras.Sequential([
        keras.layers.Input(shape=(v.split,)),
        keras.layers.Dense(units=50, activation="relu"),
        keras.layers.Dense(units=70, activation="relu"),
        keras.layers.Dense(units=20, activation="relu"),
        keras.layers.Dense(units=100, activation="linear", name="outputs"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0015275),
                  loss="mse",
                  metrics=["accuracy"])
    history = model.fit(x_train, y_train,
                        epochs=100,
                        validation_split=0.2,
                        verbose=1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss over training')
    plt.ylabel('Loss (MSE)')
    plt.xlabel('Epoch count')
    plt.legend(['Training data', 'Testing data'], loc='upper left')
    plt.show()
    model.save(f'models/{v.filename}_model.keras')
    model.summary()


def run_keras(x_train, y_train) -> None:
    def build_model(hp):
        model = keras.Sequential()
        model.add(keras.layers.Input(shape=(v.split,)))
        # Tune the number of layers.
        n_units = hp.Int("n_units", min_value=1, max_value=4, step=1)
        for i in range(n_units):
            model.add(
                keras.layers.Dense(
                    # Tune number of units separately.
                    units=hp.Int(f"units_{i + 1}", min_value=10, max_value=80, step=10),
                    activation="relu"
                )
            )
        model.add(keras.layers.Dense(100, activation="linear"))
        learning_rate = hp.Float("lr", min_value=1e-5, max_value=1e-1, sampling="log")
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
        return model

    tuner = keras_tuner.BayesianOptimization(
        hypermodel=build_model,
        objective="val_loss",
        max_trials=2000
    )

    tuner.search(x_train, y_train,
                 epochs=20, validation_split=0.2)


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


initiate(0.2, False)
