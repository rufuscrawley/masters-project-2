import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Normalization
from keras.src.optimizers import Adam
from sklearn.model_selection import train_test_split

df = pd.read_csv('outputs.csv')

input_data, output_data = df.iloc[:, :15], df.iloc[:, 15:]

combined_batch = tf.constant(np.expand_dims(np.stack([input_data]), axis=-1), dtype=tf.float32)
normalization_layer = Normalization()
normalization_layer.adapt(combined_batch)

input_train, input_test, output_train, output_test = train_test_split(input_data, output_data, test_size=0.3,
                                                                      random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(25, activation=tf.nn.relu, input_shape=(15,)),  # input shape required
    tf.keras.layers.Dense(50, activation=tf.nn.softmax),
    tf.keras.layers.Dense(100)
])

adam = Adam(0.0001)
model.compile(optimizer=adam,
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(input_train, output_train, epochs=10)
test_loss, test_acc = model.evaluate(input_test, output_test, verbose=2)
