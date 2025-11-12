import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Read in our csv file
print("Reading .csv file...")
data = pd.read_csv('datasets/normalised.csv')
# Split it into I/O form
x, y = data.iloc[:, :14], data.iloc[:, 14:]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

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
learning_rate = 0.0001
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=learning_rate),
    loss="mse",
    metrics=["accuracy", "mae"],
)

print("Building model...")
model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))

model.save('models/final_model.keras')  # The file needs to end with the .keras extension
