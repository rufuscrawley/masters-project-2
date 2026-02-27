import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np

import utilities
import variables

import variables as v

n_consts = utilities.get_y_consts(variables.split)

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
        new_inputs.append(np.log10(value) / (n_consts[n] * -1))
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
        new_fluxes.append(np.log10(value) / (n_consts[idx] * -1))
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
        new_outputs.append(np.pow(10, value * n_consts[n] * -1))
    return new_outputs
