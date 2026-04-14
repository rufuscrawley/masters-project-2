import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import keras
import variables as v


def run_model(x_train, y_train) -> None:
    model = keras.Sequential([
        keras.layers.Input(shape=(v.split,)),
        keras.layers.Dense(units=64, activation="relu"),
        keras.layers.Dense(units=64, activation="relu"),
        keras.layers.Dense(units=100, activation="linear", name="outputs"),
    ])
    model.compile(optimizer="adam",
                  loss="mse",
                  metrics=["accuracy"])
    model.fit(x_train, y_train,
              epochs=50,
              validation_split=0.2,
              verbose=1)
    model.save(f'models/{v.filename}_model.keras')
    model.summary()
