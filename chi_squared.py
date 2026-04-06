import os

from matplotlib import pyplot as plt

import variables as v
import network_variables as nv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd
import random

random.seed()

data = pd.read_csv(v.n_file)


def graph_test(tests):
    for i in range(tests):
        # Gather training data
        ROW: int = random.randint(1, 1_000)
        x, y = data.iloc[:, :v.split], data.iloc[:, v.split:]
        x_row, y_row = (np.array(x.iloc[ROW]),
                        np.array(y.iloc[ROW]))

        print(f"Feeding network {x_row}!")
        results = nv.predict(x_row)
        print(results)
        plt.plot(v.wavelengths, y_row)
        plt.plot(v.wavelengths, (results), label=f"guess {i + 1}")
        plt.xscale("log")
        plt.yscale("log")

    plt.legend()
    plt.show()


graph_test(1)
