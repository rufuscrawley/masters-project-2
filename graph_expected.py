import os

from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np

import conversion
import genetic_algorithm
import utilities
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

cs = make_interp_spline(lupi_wavelengths, vals.convert_to_si(), k=3)

wavelengths_interpolated = cs(variables.wavelengths)
n_consts = utilities.get_y_consts(variables.split)

# for n, wavelength in enumerate(wavelengths_interpolated):
#     print(f"wavelength = {wavelength}")
#     new_wavelength = 0 if wavelength <= 0 else np.log10(wavelength)
#     wavelengths_interpolated[n] = (new_wavelength * -1) / n_consts[n]

# [0.19158496 0.26132977 0.18411773 0.14736826 0.14690377 0.16708187
#  0.13157677 0.16461835 0.17053315 0.23383445 0.3381894  0.22613224
#  0.16264602 0.23910073 0.17646876 0.18375153 0.14706464 0.18451725
#  0.15576799 0.18265436 0.2289103  0.13798782 0.15866816 0.14537254
#  0.18895675 0.16354925 0.24390956 0.1542448  0.22838145 0.15722318
#  0.22475426 0.1916079  0.1651851  0.25950835 0.20838096 0.21931901
#  0.18974747 0.33135888 0.15973582 0.16939207 0.16482107 0.16723494
#  0.20350526 0.26351039 0.35497736 0.16256549 0.19907605 0.15971591
#  0.22236427 0.22401685 0.19403064 0.17914859 0.29510959 0.33203602
#  0.26619267 0.26962491 0.22332892 0.38487822 0.48020716 0.45777948
#  0.43393318 0.62881755 0.58850157 0.59186744 0.63879823 0.61623753
#  0.65235777 0.65137556 0.65520457 0.65870529 0.65205463 0.65698317
#  0.66173675 0.66788044 0.67256335 0.677975   0.68283563 0.68862769
#  0.69343685 0.69815654 0.70293688 0.70695288 0.7104602  0.71391633
#  0.71759997 0.72030667 0.72457822 0.72781163 0.73018972 0.7342239
#  0.73786329 0.74166424 0.74429332 0.7477036  0.7520204  0.75599764
#  0.76016474 0.76287362 0.76668654 0.77302775]


#

model_guesses = variables.model.predict(np.array([[0.03, 3000, 50,
                                                   400, 10, 1.15,
                                                   10e-3, 3.0, 3900]]),
                                        verbose=0)[0]
better_guesses = genetic_algorithm.denormalise_outputs(model_guesses)

# solution = genetic_algorithm.find_solution(wavelengths_interpolated)
# genetic_algorithm.graph_outputs(solution, vals)
plt.title("Observed SED")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Wavelength (Microns)")
plt.ylabel("Flux (erg / s / cm^2)")
plt.show()
