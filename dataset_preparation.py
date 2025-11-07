import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras
import keras_tuner
import numpy as np
import pandas as pd
from keras import metrics
from sklearn.model_selection import train_test_split


def normalise(self, column, log=False, extra_divisor=1):
    if log:
        self[column] = self[column].map(lambda value: 999 if value <= 0 else np.log10(value))
    min_val = self[column].min()
    max_val = self[column].max()
    if max_val == np.inf:
        max_val = 0
    range_val = max_val - min_val
    if range_val != 0:
        self[column] = self[column].map(lambda value: value / (range_val * extra_divisor))
    else:
        self[column] = self[column].map(lambda value: value / extra_divisor)


# Oo oo aa aa monkey patch madness
pd.DataFrame.normalise = normalise

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

# Read in our csv file
print("Reading .csv file...")
data = pd.read_csv('datasets/outputs.csv')
data = data.drop('ninc', axis=1)
# Split it into I/O form
x, y = data.iloc[:, :14], data.iloc[:, 14:]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# Normalise the training set
print("Normalising dataset...")
x_train.normalise("amin1", True, -1.2)
x_train.normalise("amax1", True, 3)
x_train.normalise("inclinations")
x_train.normalise("Stellar_age", True, 3)
x_train.normalise("mass1")
x_train.normalise("Temp_sublimation", extra_divisor=1599)
x_train.normalise("router", extra_divisor=2)
x_train.normalise("height")
x_train.normalise("betadisc", extra_divisor=3)
x_train.normalise("alphadisc")
x_train.normalise("mdisc")
x_train.normalise("Stellar_radius")
x_train.normalise("Stellar_temperature")
x_train.normalise("rinner")
print("Normalising Y values...")
for index, row in enumerate(y_train):
    y_train.normalise(row, True, -1)

print("Stitching DataFrames...")
df = x_train.join(y_train)
df.to_csv('datasets/normalised.csv', index=False)

# Create the validation set
print("Creating validation set...")
training_length = len(x_train)
training_ratio = 0.8

print(
    f"Splitting {training_length} datapoints - training set gets {np.floor(training_length * training_ratio)},"
    f" validation gets {np.floor(training_length * (1 - training_ratio))}")
split_point = int(np.floor(training_length * training_ratio))
x_val = x_train[-split_point:]
y_val = y_train[-split_point:]
x_train = x_train[:-split_point]
y_train = y_train[:-split_point]

# Normalisation test 2?
print("Creating normalisation layer...")
normalize = keras.layers.Normalization()
normalize.adapt(x_train.to_numpy())

print("Setting up the model...")


def build_model(hp):
    model = keras.Sequential()
    model.add(normalize)
    model.add(keras.layers.Flatten())
    # Tune the number of layers.
    for i in range(hp.Int("num_layers", 2, 6)):
        model.add(
            keras.layers.Dense(
                # Tune number of units separately.
                units=hp.Int(f"units_{i}", min_value=64, max_value=640, step=64),
                activation=hp.Choice("activation", ["relu"]),
            )
        )
    if hp.Boolean("dropout"):
        model.add(keras.layers.Dropout(rate=0.25))
    model.add(keras.layers.Dense(100, activation="softmax"))
    learning_rate = hp.Float("lr", min_value=1e-5, max_value=1e-3, sampling="log")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["accuracy"],
    )
    return model


print("Building model...")
tuner = keras_tuner.RandomSearch(
    hypermodel=build_model,
    objective="val_loss",
    max_trials=1000,
    executions_per_trial=2,
)

tuner.search(x_train, y_train,
             epochs=10,
             validation_data=(x_val, y_val))
