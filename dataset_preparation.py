import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import keras
import keras_tuner
import numpy as np
import pandas as pd
from keras import layers
from sklearn.model_selection import train_test_split

# Catch errors in normalise()
np.seterr(all='raise')


def normalise(self, column, log=False):
    if log:
        self[column] = self[column].map(lambda age: 0 if age == 0 else np.log10(age))
        min_val = self[column].min()
        max_val = self[column].max()
        range_val = max_val - min_val
    else:
        min_val = self[column].min()
        max_val = self[column].max()
        range_val = max_val - min_val
        self[column] = self[column].map(lambda age: age / range_val)


# Oo oo aa aa monkey patch madness
pd.DataFrame.normalise = normalise

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

# Read in our csv file
data = pd.read_csv('datasets/outputs.csv')

# Split it into I/O form
x, y = data.iloc[:, :15], data.iloc[:, 15:]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# Normalise the training set
x_train.normalise("amin1", True)
x_train.normalise("amax1", True)
x_train.normalise("inclinations")
x_train.normalise("Stellar_age", True)
x_train.normalise("mass1")
x_train.normalise("Temp_sublimation")
x_train.normalise("router")
x_train.normalise("height")
x_train.normalise("betadisc")
x_train.normalise("alphadisc")
x_train.normalise("mdisc")
x_train.normalise("Stellar_radius")
x_train.normalise("Stellar_temperature")
x_train.normalise("rinner")
for row in y_train:
    y_train.normalise(row, True)

# Create the validation set
split_point = 30000
x_val = x_train[-split_point:]
y_val = y_train[-split_point:]
x_train = x_train[:-split_point]
y_train = y_train[:-split_point]

# Normalisation test 2?
normalize = layers.Normalization()
normalize.adapt(x_train.to_numpy())


def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Flatten())
    # Tune the number of layers.
    for i in range(hp.Int("num_layers", 1, 5)):
        model.add(
            layers.Dense(
                # Tune number of units separately.
                units=hp.Int(f"units_{i}", min_value=32, max_value=512, step=32),
                activation=hp.Choice("activation", ["relu", "tanh"]),
            )
        )
    if hp.Boolean("dropout"):
        model.add(layers.Dropout(rate=0.25))
    model.add(layers.Dense(100, activation="softmax"))
    learning_rate = hp.Float("lr", min_value=1e-8, max_value=1e-1, sampling="log")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["accuracy"],
    )
    return model


tuner = keras_tuner.RandomSearch(
    hypermodel=build_model,
    objective="val_accuracy",
    max_trials=10,
    executions_per_trial=2,
)

tuner.search(x_train, y_train, epochs=10, validation_data=(x_val, y_val))
