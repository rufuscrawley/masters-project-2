import os

from testing import genetic_algorithm
import variables

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import scipy

random.seed()
reconstructed_model = variables.model

data = pd.read_csv(variables.n_file)


def graph_test(tests):
    for i in range(tests):
        ROW: int = random.randint(1, 1_000)

        x, y = data.iloc[:, :variables.split], data.iloc[:, variables.split:]
        x_row, y_row = np.array([x.iloc[ROW]]), np.array([y.iloc[ROW]])


        results = reconstructed_model.predict(x_row, verbose=0)

        spline = scipy.interpolate.CubicSpline(variables.wavelengths, y_row[0])
        true_spline = spline(variables.wavelengths)

        truest_spline = genetic_algorithm.denormalise_outputs(true_spline)
        true_nn = genetic_algorithm.denormalise_outputs(results[0])

        plt.plot(variables.wavelengths, np.pow(10, truest_spline), label=f'result_{i}')
        plt.plot(variables.wavelengths, np.pow(10, true_nn), label="pred")

    plt.title("Observed SED")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Wavelength (Microns)")
    plt.ylabel("Flux (erg / s / cm^2)")
    plt.legend()
    plt.show()


graph_test(1)
