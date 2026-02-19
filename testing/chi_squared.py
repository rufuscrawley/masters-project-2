import os

import variables
from libraries.pycubicspline import Spline

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

random.seed()
reconstructed_model = keras.models.load_model("../models/outputs_lupi_model.keras")

data = pd.read_csv('../datasets/outputs_lupi_test.csv')


def graph_test(tests):
    for i in range(tests):
        ROW = random.randint(1, 1_000)

        x, y = data.iloc[:, :13], data.iloc[:, 13:]
        x_row, y_row = x.iloc[ROW], y.iloc[ROW]

        y_row = np.array([y_row])
        results = reconstructed_model.predict(np.array([x_row]), verbose=0)

        rx = np.arange(-2, 1300, 1)

        spline = Spline(variables.wavelengths, y_row[0])
        ry = [spline.calc(i) for i in rx]
        plt.plot(rx, ry, label=f'result_{i}')

        spline_pred = Spline(variables.wavelengths, results[0])
        ry_pred = [spline_pred.calc(i) for i in rx]
        plt.plot(rx, ry_pred, label=f'pred result_{i}')

    plt.grid(True)
    plt.legend()
    plt.show()


graph_test(5)
