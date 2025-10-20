import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import numpy as np
from keras import layers

from sklearn.model_selection import train_test_split
import keras_tuner
import keras

np.seterr(all='raise')


def normalise(self, column, log=False):
    if log:
        try:
            self[column] = self[column].map(lambda age: np.log10(age))
            min_val = self[column].min()
            max_val = self[column].max()
            range_val = max_val - min_val
            self[column] = self[column].map(lambda age: age / range_val)
        except:
            self[column] = 0
    else:
        min_val = self[column].min()
        max_val = self[column].max()
        range_val = max_val - min_val
        self[column] = self[column].map(lambda age: age / range_val)


pd.DataFrame.normalise = normalise

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

# Read in our csv file
data = pd.read_csv('datasets/outputs.csv')

# Split it into I/O form
x, y = data.iloc[:, :15], data.iloc[:, 15:]

# Normalise the training set

x.normalise("amin1", True)
x.normalise("amax1", True)
x.normalise("inclinations")
x.normalise("Stellar_age", True)
x.normalise("mass1")
x.normalise("Temp_sublimation")
x.normalise("router")
x.normalise("height")
x.normalise("betadisc")
x.normalise("alphadisc")
x.normalise("mdisc")
x.normalise("Stellar_radius")
x.normalise("Stellar_temperature")
x.normalise("rinner")

for row in y:
    y.normalise(row, True)

y = np.array(y)
x = np.array(x)



# Split it into training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.3,
                                                    random_state=42)

# Split the training sets into validation sets
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]


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
    model.add(layers.Dense(100, activation="relu"))
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


tuner = keras_tuner.RandomSearch(
    hypermodel=build_model,
    objective="val_accuracy",
    max_trials=3,
    executions_per_trial=2,
)

tuner.search(x_train, y_train, epochs=10, validation_data=(x_val, y_val))
