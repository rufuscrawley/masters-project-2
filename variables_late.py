import warnings

import astropy.units as u
from scipy.interpolate import CubicSpline, PchipInterpolator

# Suppress only UserWarnings from the module
warnings.filterwarnings("ignore", category=UserWarning)
import keras
import numpy as np
import pandas as pd
import tensorflow as tf

import variables_early as ve

model = keras.models.load_model(f"models/{ve.filename}_model.keras", safe_mode=False)
input_spec = tf.TensorSpec(shape=[None, ve.split], dtype=tf.float32)
call_model = tf.function(model, input_signature=[input_spec], reduce_retracing=True)

output_consts = pd.read_csv(ve.const_file)
x_consts = np.array(output_consts[:ve.split])
mean_x, std_x = x_consts[:, 0], x_consts[:, 1]
y_consts = np.array(output_consts[ve.split:])
mean_y, std_y = y_consts[:, 0], y_consts[:, 1]

logs = []
for log in ve.included:
    logs.append(ve.names[log])


def denormalise(col, mean, std, log_norm=True):
    col = (col * std) + mean
    if log_norm:
        col = np.pow(10, col)
    return col


def normalise(val, mean, std, log_norm=False):
    if log_norm:
        val = np.log10(val)
    return (val - mean) / std


def predict_fluxes(input_data: np.ndarray, normalised=False) -> np.ndarray:
    """
    Predicts flux values for given stellar parameters.
    :param normalised: Whether the input parameters are normalised.
    :param input_data: Can be a list of lists, a numpy array, or a pandas DataFrame. Example: [[1.3, 1.6, 5800, 1500], ...]
    :return: A numpy array of predicted flux values (shape: samples x 100).
    """
    if not normalised:
        norm_xs = []
        for index, inputs in enumerate(input_data):
            s = normalise(inputs, mean_x[index], std_x[index], logs[index])
            norm_xs.append(s)
        norm_xs = np.array(norm_xs)
    else:
        norm_xs = input_data

    new_x = tf.convert_to_tensor(norm_xs.flatten())
    predictions_scaled = call_model([new_x])
    predictions_flux = denormalise(predictions_scaled, mean_y, std_y)[0]

    return predictions_flux


def interpolate_fluxes(fluxes, wavelengths) -> np.ndarray:
    """
    Interpolates 100 fluxes from TORUS into `n_interpolate` fluxes using a cubic spline, then returns
    them over a set of predefined wavelengths.
    :param fluxes:
    :param wavelengths:
    :return:
    """
    spline = PchipInterpolator(ve.wavelengths, fluxes, extrapolate=False)
    true_spline = spline(wavelengths)

    true_spline = np.clip(true_spline, 1e-24, None)
    return true_spline


def apply_extinction(fluxes, a_v) -> None:
    fluxes[:ve.n_finish] *= ve.extmod.extinguish(ve.wavelengths[:ve.n_finish]
                                                 * u.micron,
                                                 Av=a_v)
