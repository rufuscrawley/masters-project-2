import os

import pandas as pd
from scipy.interpolate import CubicSpline

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np

import utilities
import variables

import variables as v

x_consts = utilities.get_x_consts(variables.split)
y_consts = utilities.get_y_consts(variables.split)


######################
# FLUX NORMALISATION #
######################

def normalise_fluxes(fluxes):
    """
    Receives a set of un-normalised fluxes in erg/s/cm^-2
    and normalises them to be fed to the neural network.
    :return:
    """
    new_inputs = []
    for n, value in enumerate(fluxes):
        if value <= 0 and n != 0:
            new_inputs.append(new_inputs[n - 1])
            continue
        new_inputs.append(np.log10(value) / (y_consts[n] * -1))
    return new_inputs


def normalise_uneven_fluxes(fluxes, wavelengths):
    new_fluxes = []
    wavs = np.array(wavelengths)
    for n, value in enumerate(fluxes):
        # Find index of the closest wavelength that we are normalising
        wavelength_target = wavs[n]
        idx = np.abs(v.wavelengths - wavelength_target).argmin()
        if value <= 0 and n != 0:
            print("Oh dear!")
            new_fluxes.append(new_fluxes[n - 1])
            continue
        new_fluxes.append(np.log10(value) / (y_consts[idx] * -1))
    return new_fluxes


def denormalise_fluxes(solution) -> list:
    """
    Denormalises the solution parameters outputted by the genetic algorithm.
    :return:
    """
    new_outputs = []
    for n, value in enumerate(solution):
        if value == 0 and n != 0:
            new_outputs.append(new_outputs[n - 1])
            continue
        new_outputs.append(np.pow(10, value * y_consts[n] * -1))
    return new_outputs


def interpolate_fluxes(fluxes, wavelengths):
    """
    Interpolates 100 fluxes from TORUS into `n_interpolate` fluxes using a cubic spline, then returns
    them over a set of predefined wavelengths.
    :param fluxes:
    :param wavelengths:
    :return:
    """
    spline = CubicSpline(v.wavelengths, fluxes, extrapolate=False)
    true_spline = spline(wavelengths)
    return true_spline


def normalise_inputs(inputs):
    solution, i = [], 0
    for key in v.names.keys():
        if key in variables.excluded: continue
        invert = -1 if v.names[key]["invert"] else 1
        if v.names[key]["logarithmic"]:
            result = np.log10(inputs[i])
        else:
            result = inputs[i]
        n_input = result * invert / x_consts[i]
        solution.append(n_input)
        i += 1
    return solution


def denormalise_inputs(solution):
    inputs = []
    i = 0
    for key in v.names.keys():
        if key in variables.excluded: continue
        invert = -1 if v.names[key]["invert"] else 1
        n_solution = solution[i] * invert * x_consts[i]
        if v.names[key]["logarithmic"]:
            result = np.pow(10, n_solution)
            inputs.append(result)
            print(f"{key} = {result}")
        else:
            inputs.append(n_solution)
            print(f"{key} = {n_solution}")
        i += 1
    return inputs
