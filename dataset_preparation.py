import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras
import keras_tuner
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def normalise(self, column, log=False, invert=False):
    if log:
        self[column] = self[column].map(lambda value: 0 if value <= 0 else np.log10(value))
    max_val = self[column].abs().max()
    self[column] = self[column].map(lambda value: value / max_val)
    if invert:
        self[column] = self[column].map(lambda value: value * -1)


pd.DataFrame.normalise = normalise

# Read in our csv file
print("Reading .csv file...")
data = pd.read_csv('datasets/outputs.csv')
data = data.drop('ninc', axis=1)

# Drop any inf values
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)

# Split it into I/O form
x, y = data.iloc[:, :14], data.iloc[:, 14:]
print("x:")
print(x)
print("y:")
print(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Normalise the training set
print("Normalising dataset...")
x_train.normalise("amin1", True, True)
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
print("Normalising Y values...")
for row in y_train:
    y_train.normalise(row, True, True)

print("Stitching DataFrames...")
df = x_train.join(y_train)
df.to_csv('datasets/normalised.csv', index=False)

# Create the validation set
print("Creating validation set...")
training_length = len(x_train)
training_ratio = 0.2
print(
    f"Splitting {training_length} datapoints - validation set gets {np.floor(training_length * training_ratio)},"
    f" training gets {np.floor(training_length * (1 - training_ratio))}")
split_point = int(np.floor(training_length * training_ratio))
x_val, y_val = x_train[-split_point:], y_train[-split_point:]
x_train, y_train = x_train[:-split_point], y_train[:-split_point]

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
    for i in range(4):
        model.add(
            keras.layers.Dense(
                # Tune number of units separately.
                units=hp.Int(f"units_{i}", min_value=100, max_value=400, step=25),
                activation="relu",
            )
        )
    model.add(keras.layers.Dense(100, activation="softmax"))
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mean_absolute_error",
        metrics=["mse"],
    )
    return model


print("Building model...")
tuner = keras_tuner.RandomSearch(
    hypermodel=build_model,
    objective="val_loss",
    max_trials=1000,
    executions_per_trial=2,
)

tuner.search(x_train, y_train, epochs=10, validation_data=(x_val, y_val))
