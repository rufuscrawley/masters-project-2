import os

import variables

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import scipy

random.seed()
reconstructed_model = keras.models.load_model("../models/outputs_model.keras")

data = pd.read_csv('../datasets/normalised/n_outputs.csv')


def graph_test(tests):
    for i in range(tests):

        ROW: int = random.randint(1, 1_000)

        x, y = data.iloc[:, :13], data.iloc[:, 13:]

        x_row, y_row = np.array([x.iloc[ROW]]), np.array([y.iloc[ROW]])

        results = reconstructed_model.predict(x_row, verbose=0)



        spline = scipy.interpolate.CubicSpline(variables.wavelengths, y_row[0])
        plt.plot(variables.wavelengths, spline(variables.wavelengths), label=f'result_{i}')
        plt.plot(variables.wavelengths, results[0], label="pred")

    plt.grid(True)
    plt.title("Observed SED")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Wavelength (Microns)")
    plt.ylabel("Flux (erg / s / cm^2)")
    plt.legend()
    plt.show()


graph_test(1)
