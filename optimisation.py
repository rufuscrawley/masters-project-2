import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras
import keras_tuner


def run_keras(x_train, y_train, x_val, y_val):
    # Normalisation test 2?
    print("Creating normalisation layer...")
    normalize = keras.layers.Normalization()
    normalize.adapt(x_train.to_numpy())

    print("Setting up the model...")

    def build_model(hp):
        model = keras.Sequential()
        model.add(normalize)
        model.add(keras.layers.Flatten())
        # Tune the number of layers.
        for i in range(hp.Int("layers", min_value=1, max_value=5)):
            model.add(
                keras.layers.Dense(
                    # Tune number of units separately.
                    units=hp.Int(f"units_{i}", min_value=50, max_value=400, step=25),
                    activation="relu",
                )
            )
        model.add(keras.layers.Dense(100, activation="relu"))
        learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="mse",
            metrics=["accuracy"],
        )
        return model

    print("Building model...")
    tuner = keras_tuner.RandomSearch(
        hypermodel=build_model,
        objective="val_loss",
        max_trials=1000,
        executions_per_trial=2,
    )

    tuner.search(x_train, y_train, epochs=10, validation_data=(x_val, y_val))


def run_model(x_train, y_train, x_val, y_val, file_path):
    print("Creating normalisation layer...")
    normalize = keras.layers.Normalization()
    normalize.adapt(x_train.to_numpy())

    model = keras.Sequential()
    model.add(normalize)
    model.add(keras.layers.Flatten())
    # Tune the number of layers.
    model.add(keras.layers.Dense(units=100, activation="relu"))
    model.add(keras.layers.Dense(units=175, activation="relu"))
    model.add(keras.layers.Dense(375, activation="relu"))
    model.add(keras.layers.Dense(units=100, activation="relu"))
    learning_rate = 0.00063953
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=learning_rate),
        loss="mse",
        metrics=["accuracy"],
    )

    print("Building model...")
    model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))

    model.save(f'models/{file_path}_model.keras')
