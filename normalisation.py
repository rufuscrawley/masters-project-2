import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd


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
# Randomise rows
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

print("Stitching DataFrames...")
df = x.join(y)
df.to_csv('datasets/normalised.csv', index=False)
