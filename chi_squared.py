import os

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

import variables_early as v
import variables_late as nv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd
import random

random.seed()

data = pd.read_csv(v.test_file)


def plot_residues():
    x, y = data.iloc[:, :v.split], data.iloc[:, v.split:]
    residue_list, mean_list, std_list = [], [], []
    length = data.shape[0]
    skips = 0
    for i in range(length):
        # Gather training data
        ROW = i
        x_row, y_row = (np.array(x.iloc[ROW]), np.array(y.iloc[ROW]))
        y_row = nv.denormalise(y_row, nv.mean_y, nv.std_y)
        results = nv.predict_fluxes(x_row, True)

        residues = (np.log10(y_row) - np.log10(results)) / np.log10(y_row)

        if any(map((-.04).__gt__, residues)) or any(map(.04.__lt__, residues)):
            skips += 1
        else:
            residue_list.append(residues)

    residue_list = (np.array(residue_list) * 100).transpose()

    means = np.mean(residue_list, axis=1)
    std = np.std(residue_list, axis=1)

    residue_list = residue_list.flatten()
    wavelengths = np.repeat(v.wavelengths, length - skips)

    plt.hist2d(wavelengths, residue_list,
               bins=100, norm=LogNorm(),
               density=False, cmin=8)
    plt.plot(v.wavelengths, means, color="k", label="$\mu$")
    plt.plot(v.wavelengths, (means + std), color="orange", label="1$\sigma$", linestyle="dashed")
    plt.plot(v.wavelengths, (means - std), color="orange", linestyle="dashed")
    plt.plot(v.wavelengths, (means + (2 * std)), color="r", label="2$\sigma$", linestyle="dashed")
    plt.plot(v.wavelengths, (means - (2 * std)), color="r", linestyle="dashed")
    plt.title("NN Residuals against test dataset")
    plt.xlabel("Wavelength ($\mu$m)")
    plt.ylabel("Relative residual (%)")
    plt.tight_layout()
    plt.grid()
    plt.legend()
    plt.show()

    plt.plot(v.wavelengths, means, color="k", label="$\mu$")
    plt.plot(v.wavelengths, (means + std), color="orange", label="1$\sigma$", linestyle="dashed")
    plt.plot(v.wavelengths, (means - std), color="orange", linestyle="dashed")
    plt.plot(v.wavelengths, (means + (2 * std)), color="r", label="2$\sigma$", linestyle="dashed")
    plt.plot(v.wavelengths, (means - (2 * std)), color="r", linestyle="dashed")
    plt.title("NN Means/StDev against test dataset")
    plt.xlabel("Wavelength ($\mu$m)")
    plt.ylabel("Relative residual (%)")
    plt.tight_layout()
    plt.grid()
    plt.xscale("log")
    plt.legend()
    plt.show()


def plot_comparisons():
    x, y = data.iloc[:, :v.split], data.iloc[:, v.split:]

    fig, axs = plt.subplots(3, 3)
    fig.suptitle("Example SED fits against training data set")
    fig.supxlabel("Wavelength ($\mu$m)")
    fig.supylabel("$\lambda$F (erg / s / cm$^2$)")
    for i in range(3):
        for j in range(3):
            # Gather training data
            ROW: int = random.randint(1, 10_000)
            x_row, y_row = (np.array(x.iloc[ROW]), np.array(y.iloc[ROW]))

            results = nv.predict_fluxes(x_row, True)
            y_row = nv.denormalise(y_row, nv.mean_y, nv.std_y)

            axs[i, j].loglog(v.wavelengths, y_row, label="Exp.")
            axs[i, j].loglog(v.wavelengths, results, label="Pred.")
    plt.tight_layout()
    plt.legend()
    plt.show()


plot_comparisons()
plot_residues()
