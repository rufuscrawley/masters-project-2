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
y_consts = np.array(output_consts[v.split:])
y_consts = y_consts.flatten()
logs = []
for log in v.included:
    logs.append(v.names[log])


def denormalise(col, consts, log=True):
    col = col * consts
    if log:
        col = np.pow(10, col)
    return col


def normalise(val, const, log=False):
    if log:
        val = np.log10(val)
    val = val / const
    return val


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
            s = normalise(inputs, x_consts[index], logs[index])
            norm_xs.append(s)
        norm_xs = np.array(norm_xs)
    else:
        norm_xs = input_data

    new_x = tf.convert_to_tensor(norm_xs.flatten())
    predictions_scaled = call_model([new_x])
    predictions_flux = denormalise(predictions_scaled, y_consts)

    return predictions_flux[0]
