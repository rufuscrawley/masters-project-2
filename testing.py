import keras_tuner
import keras


def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.Dense(
        hp.Choice('units', [8, 16, 32]),
        activation='relu'))
    model.add(keras.layers.Dense(1, activation='relu'))
    model.compile(loss='mse')
    return model
