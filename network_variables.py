import warnings
from typing import Any

# Suppress only UserWarnings from the module
warnings.filterwarnings("ignore", category=UserWarning)
import keras
import numpy as np
import pandas as pd
import tensorflow as tf

import variables as v

model = keras.models.load_model(f"models/{v.filename}_model.keras", safe_mode=False)
input_spec = tf.TensorSpec(shape=[None, v.split], dtype=tf.float32)
call_model = tf.function(model, input_signature=[input_spec], reduce_retracing=True)

output_consts = pd.read_csv(v.const_file)
x_consts = np.array(output_consts[:v.split])
mean_x, std_x = x_consts[:, 0], x_consts[:, 1]
y_consts = np.array(output_consts[v.split:])
mean_y, std_y = y_consts[:, 0], y_consts[:, 1]

logs = []
for log in v.included:
    logs.append(v.names[log])


def denormalise(col, mean, std, log_norm=True):
    col = (col * std) + mean
    if log_norm:
        col = np.pow(10, col)
    return col


def normalise(val, mean, std, log_norm=False):
    if log_norm:
        val = np.log10(val)
    return (val - mean) / std


def predict_fluxes(input_data: np.ndarray, normalised=False) -> Any:
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
