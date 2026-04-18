# Central file to collate all relevant functions

import pipeline
import utilities
import mcmc
import genetic_algorithm as ga

wavelengths = [0.545, 0.638, 0.797,
               1.22, 1.63, 2.2,
               3.6, 4.5, 5.8,
               8.0, 24, 61.1,
               70, 74.8, 89.3]
fluxes = [0.0655, 0.120, 0.216,
          0.483, 0.591, 0.511,
          0.324, 0.220, 0.313,
          0.370, 0.765, 1.42,
          1.581, 1.480, 1.260]
janksys = True
if janksys:
    fluxes = utilities.JanskyWavelengths(fluxes, wavelengths).convert_to_si()
parameters = wavelengths, fluxes

# Sets up the neural network
# pipeline.initiate(0.2, False)
best_solution = ga.run(parameters, 5, 25_000)
samples = mcmc.run(parameters, best_solution, 2_500, 20)
mcmc.analyse_run(samples)
