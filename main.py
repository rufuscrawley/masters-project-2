import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.src.optimizers import Adam
from sklearn.model_selection import train_test_split
import keras_tuner
import keras

from keras import layers

# from tensorflow.keras.models import Sequential

# 15 inputs -> 100 data outputs

df = pd.read_csv('outputs.csv')
df = df.dropna()

x, y = df.iloc[:, :15], df.iloc[:, 15:]

# Now normalise axes
x = pd.DataFrame(x)
x['amin1'] = x['amin1'].map(lambda age: np.log10(age))
x['amax1'] = x['amax1'].map(lambda age: np.log10(age))
x['inclinations'] = x['inclinations'].map(lambda angle: angle / 90)
x['Stellar_age'] = x['Stellar_age'].map(lambda age: np.log10(age))
x['Temp_sublimation'] = x['Temp_sublimation'].map(lambda temperature: temperature / 1500)
x['Stellar_temperature'] = x['Stellar_temperature'].map(lambda temperature: np.log10(temperature))
x.drop(0)

y = pd.DataFrame(y)
y = y.map(lambda wavelength: 0 if wavelength == 0 else (np.log10(wavelength)))
y.drop(0)

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.3)


def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Flatten())
    # Tune the number of layers.
    for i in range(hp.Int("num_layers", 1, 3)):
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
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


print(build_model(keras_tuner.HyperParameters()))

tuner = keras_tuner.RandomSearch(
    hypermodel=build_model,
    objective="val_accuracy",
    max_trials=3,
    executions_per_trial=2,
)

tuner.search(x_train, y_train, epochs=10, validation_data=(x_val, y_val))
