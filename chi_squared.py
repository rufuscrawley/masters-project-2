import os

from matplotlib import pyplot as plt

import normalisation
import variables as v

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd
import random

random.seed()
reconstructed_model = v.model

data = pd.read_csv(v.n_file)


def graph_test():
    # Gather training data
    ROW: int = random.randint(1, 1_000)
    x, y = data.iloc[:, :v.split], data.iloc[:, v.split:]
    x_row, y_row = np.array([x.iloc[ROW]]), np.array([y.iloc[ROW]])

    results = reconstructed_model.predict(x_row, verbose=0)
    expected_results = normalisation.denormalise_fluxes(y_row[0])
    results = normalisation.denormalise_fluxes(results[0])
    residues = np.array(expected_results) - np.array(results)

    plt.hist2d(v.wavelengths, residues, bins=(100, 100), cmap='Blues')
    plt.xscale("log")
    plt.show()


graph_test()
