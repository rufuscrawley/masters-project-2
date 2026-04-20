# Central file to collate all relevant functions

import pipeline
import utilities
import mcmc
import genetic_algorithm as ga
import variables as v
import numpy as np
import astropy.units as u

wavelengths = np.array([0.545, 0.638, 0.797,
                        1.22, 1.63, 2.2,
                        3.6, 4.5, 5.8,
                        8.0, 24, 61.1,
                        70, 74.8, 89.3])
fluxes = np.array([0.0655, 0.120, 0.216,
                   0.483, 0.591, 0.511,
                   0.324, 0.220, 0.313,
                   0.370, 0.765, 1.42,
                   1.581, 1.480, 1.260])
errors = np.array([0.0007, 0.00012, 0.0022,
                   0.0048, 0.0059, 0.0051,
                   0.0184, 0.0178, 0.0156,
                   0.0223, 0.0708, 0.0220,
                   0.127, 0.37, 0.51])

janksys = True
if janksys:
    fluxes = utilities.JanskyWavelengths(fluxes, wavelengths).convert_to_si()
    errors = utilities.JanskyWavelengths(errors, wavelengths).convert_to_si()

scaling_factor = ((190 * u.pc) / v.distance_scalar) ** 2
# fluxes = fluxes * scaling_factor
parameters = wavelengths, fluxes
# print(parameters)

print("Here goes nothing...")
# Sets up the neural network
# pipeline.initiate(0.2, False)
best_solution = ga.run(parameters, errors, 5, 15_000)
samples = mcmc.run(parameters, best_solution, 2_000, 5)
mcmc.analyse_run(samples, 500)
