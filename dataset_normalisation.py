import os

from matplotlib import pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import keras
import keras_tuner
import numpy as np
import pandas as pd
from keras import layers
from sklearn.model_selection import train_test_split


# Catch errors in normalise()
# np.seterr(all='raise')


def normalise(self, column, log=False, extra_divisor=1):
    if log:
        self[column] = self[column].map(lambda value: 0 if value <= 0 else np.log10(value))
    min_val = self[column].min()
    max_val = self[column].max()
    if max_val == np.inf:
        max_val = 0
    range_val = max_val - min_val
    self[column] = self[column].map(lambda value: value / (range_val * extra_divisor))


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
    y_train.normalise(row, True)

print("Stitching DataFrames...")
df = x_train.join(y_train)
df.to_csv('datasets/normalised.csv', index=False)

# Create the validation set
print("Creating validation set...")
split_point = 40000
x_val = x_train[-split_point:]
y_val = y_train[-split_point:]
x_train = x_train[:-split_point]
y_train = y_train[:-split_point]
