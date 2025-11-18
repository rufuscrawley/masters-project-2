import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd
from variables import names


def normalise(self, column, normalisation, invert=False):
    if normalisation == "log":
        self.loc[:, column] = self[column].map(lambda value: 0 if value <= 0 else np.log10(value))
    elif normalisation == "lin":
        pass
    max_val = self[column].abs().max()
    self.loc[:, column] = self[column].map(lambda value: value / max_val)
    if invert:
        self.loc[:, column] = self[column].map(lambda value: value * -1)


def dropinf(self):
    self.replace([np.inf, -np.inf], np.nan, inplace=True)
    self.dropna(inplace=True)


pd.DataFrame.normalise = normalise
pd.DataFrame.dropinf = dropinf

# Read in our csv file
print("Reading .csv file...")
data = pd.read_csv('datasets/outputs.csv')
data = data.drop('ninc', axis=1)
# Randomise rows
data.sample(frac=1)
# Drop any inf values
data.dropinf()

# Split it into I/O form
x, y = data.iloc[:, :14], data.iloc[:, 14:]
print("Normalising dataset...")
for key in names:
    x.normalise(key, names[key][0], names[key][1])
print("Normalising Y values...")
for row in y:
    y.normalise(row, log_norm=True, invert=True)

print("Stitching DataFrames...")
df = x.join(y)
df.to_csv('datasets/normalised.csv', index=False)
