import os

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

import variables as v
import network_variables as nv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd
import random

random.seed()

data = pd.read_csv(v.test_file)


def plot_residues(tests):
    x, y = data.iloc[:, :v.split], data.iloc[:, v.split:]
    residue_list = []

    for i in range(tests):
        # Gather training data
        ROW: int = random.randint(1, 10000)
        x_row, y_row = (np.array(x.iloc[ROW]), np.array(y.iloc[ROW]))
        results = nv.predict_fluxes(x_row, True)
        y_row = nv.denormalise(y_row, nv.y_consts)
        residues = (np.log10(y_row) - np.log10(results)) / np.log10(y_row)
        residue_list.append(residues)

    residue_list = np.array(residue_list)
    residue_list = np.transpose(residue_list)
    residue_list = residue_list.flatten()
    wavelengths = np.repeat(v.wavelengths, tests)

    plt.hist2d(wavelengths, residue_list,
               bins=50, norm=LogNorm())
    plt.grid()
    plt.show()


def plot_comparisons(tests):
    x, y = data.iloc[:, :v.split], data.iloc[:, v.split:]
    for i in range(tests):
        # Gather training data
        ROW: int = random.randint(1, 1_000)

        x_row, y_row = (np.array(x.iloc[ROW]), np.array(y.iloc[ROW]))

        results = nv.predict_fluxes(x_row, True)
        y_row = nv.denormalise(y_row, nv.y_consts)

        plt.loglog(v.wavelengths, y_row, label="exp")
        plt.loglog(v.wavelengths, results, label="pred")
        plt.legend()
        plt.show()


# plot_comparisons(10)
plot_residues(1000)
