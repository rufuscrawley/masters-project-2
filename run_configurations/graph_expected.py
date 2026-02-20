import os

from matplotlib import pyplot as plt

import variables

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras
import numpy as np

file_name = "outputs"
inputs = [[0.03, 3000, 50, 10000000, 1, 1500, 400, 10, 1.15, 0.001, 3, 3900, 0.32]]

print("loading model")
model = keras.models.load_model(f"../models/{file_name}_model.keras")

# fluxes = model.predict(np.array(inputs), verbose=0)
# plt.plot(variables.wavelengths, fluxes[0], label="pred")

args = [0.0655, 0.120, 0.216,
        0.483, 0.591, 0.511,
        0.324, 0.220, 0.313,
        0.370, 0.765, 1.42,
        1.581, 1.480, 1.260,
        0.1758]

# We want to convert the *observed* SED from Jy to erg/s/cm^2
# Define SED units as [SED] -> ergs/s/cm^2
# Jy = 10^-23 * erg/s/cm^2/Hz
# Jy = 10^-23 * [SED] / Hz
# SED = 10^23 * Hz

j_fluxes = list(map(lambda arg: np.log10(arg * 10e23), variables.josh_fluxes))
arguments = list(map(lambda arg: arg * 1e6, args))

plt.plot(np.log10([0.545, 0.638, 0.797,
                   1.22, 1.63, 2.2,
                   3.6, 4.5, 5.8,
                   8.0, 24, 61.1,
                   70, 74.8, 89.3,
                   1300]), np.log10(arguments),
         label="im lupi")
plt.plot(np.log10(variables.josh_wavelengths), j_fluxes, label="josh")
plt.legend()
plt.show()
