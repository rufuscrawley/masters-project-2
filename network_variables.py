import joblib
import keras
import numpy as np
import tensorflow as tf

import variables as v

model = keras.models.load_model(f"models/{v.filename}_model.keras", safe_mode=False)
input_spec = tf.TensorSpec(shape=[None, v.split], dtype=tf.float32)
call_model = tf.function(model, input_signature=[input_spec], reduce_retracing=True)


def predict(x_new):
    new_x = tf.convert_to_tensor([x_new])[0]
    predictions_scaled = call_model(np.array([new_x]))[0]

    return predictions_scaled
