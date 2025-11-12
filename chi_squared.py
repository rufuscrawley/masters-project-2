import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras
import numpy as np

reconstructed_model = keras.models.load_model("models/final_model.keras")

data = np.array(
    [0.004801989, 3102.584102, 80.84, 9988634.767, 1.314121742, 1499.861153, 149.982807, 16.6475276, 1.092703572,
     1.934854579, 0.005072641, 1.615735905, 5887.676005, 0.101557187])

results = reconstructed_model.predict(
    np.expand_dims(data, 1)
)


print(results)