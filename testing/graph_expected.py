import os

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import *

import conversion
from testing import genetic_algorithm as ga
import variables

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

lupi_wavelengths = [0.545, 0.638, 0.797,
                    1.22, 1.63, 2.2,
                    3.6, 4.5, 5.8,
                    8.0, 24, 61.1,
                    70, 74.8, 89.3, 1300]

janskys = [0.0655, 0.120, 0.216,
           0.483, 0.591, 0.511,
           0.324, 0.220, 0.313,
           0.370, 0.765, 1.42,
           1.581, 1.480, 1.260, 0.1758]

vals = conversion.JanskyWavelengths(janskys, lupi_wavelengths)


def interpolate_janskys(jansky_microns):
    spline = PchipInterpolator(np.log10(jansky_microns.wavelengths), jansky_microns.janskys, extrapolate=True)
    true_spline = spline(np.log10(variables.wavelengths))
    return conversion.JanskyWavelengths(true_spline, variables.wavelengths)


interpolated_si = interpolate_janskys(vals).convert_to_si()

normalised_si = ga.normalise_inputs(interpolated_si)

solution = ga.find_solution(normalised_si)

outputs_pred = variables.model.predict(np.array([solution]), verbose=0)[0]
outputs_pred = ga.denormalise_outputs(outputs_pred)
#
plt.plot(variables.wavelengths, outputs_pred, label="expected")
plt.scatter(vals.wavelengths, np.log10(vals.convert_to_si()))
plt.plot(variables.wavelengths, np.log10(interpolated_si), label="interpolated")
#
# plt.scatter(vals.wavelengths, np.log10(vals.janskys))

plt.title("Observed SED")
plt.xlabel("Wavelength (Microns)")
plt.ylabel("Flux (erg / s / cm^2)")
plt.xscale("log")
plt.legend()
plt.show()
