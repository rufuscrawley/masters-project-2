import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import keras
import variables


def run_model(x_train, y_train) -> None:
    normalize = keras.layers.Normalization()
    normalize.adapt(x_train.to_numpy())

    model = keras.Sequential([
        normalize,
        keras.layers.Dense(units=256, activation="leaky_relu"),
        keras.layers.Dense(units=512, activation="leaky_relu"),
        keras.layers.Dense(units=512, activation="leaky_relu"),
        keras.layers.Dense(units=256, activation="leaky_relu"),
        keras.layers.Dense(units=100, activation="linear", name="outputs"),
    ])
    learning_rate = 0.01
    model.compile(optimizer=keras.optimizers.AdamW(learning_rate=learning_rate),
                  loss="mse",
                  metrics=["mae", "accuracy"])
    model.fit(x_train, y_train,
              epochs=10,
              validation_split=0.2,
              batch_size=32,
              verbose=1)
    model.save(f'models/{variables.filename}_model.keras')
    model.summary()
