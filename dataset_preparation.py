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
data.sample(frac=1)

# Drop any inf values
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)

# Split it into I/O form
x, y = data.iloc[:, :14], data.iloc[:, 14:]
print("Normalising dataset...")
x.normalise("amin1", True, True)
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
print("Normalising Y values...")
for row in y:
    y.normalise(row, True, True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

print("Stitching DataFrames...")
df = x_train.join(y_train)
df.to_csv('datasets/normalised.csv', index=False)

# Create the validation set
print("Creating validation set...")
training_length = len(x_train)
training_ratio = 0.2
print(f"Splitting {training_length} datapoints - "
      f"validation set gets {np.floor(training_length * training_ratio)},"
      f" training gets {np.floor(training_length * (1 - training_ratio))}")
split_point = int(np.floor(training_length * training_ratio))
x_val, y_val = x_train[-split_point:], y_train[-split_point:]
x_train, y_train = x_train[:-split_point], y_train[:-split_point]

print("Setting up the model...")

# Normalisation test 2?
print("Creating normalisation layer...")
normalize = keras.layers.Normalization()
normalize.adapt(x_train.to_numpy())

model = keras.Sequential()
model.add(normalize)
model.add(keras.layers.Flatten())
# Tune the number of layers.
for i in range(4):
    model.add(
        keras.layers.Dense(
            # Tune number of units separately.
            units=100,
            activation="relu",
        )
    )
model.add(keras.layers.Dense(100, activation="softmax"))
learning_rate = 0.00189
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=learning_rate),
    loss="mse",
    metrics=["accuracy", "mae"],
)

print("Building model...")
model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))

model.save('models/final_model.keras')  # The file needs to end with the .keras extension
