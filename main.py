# Central file to collate all relevant functions

import fit_targets as ft
import genetic_algorithm as ga
import mcmc

target = ft.IMLupi

print("Here goes nothing...")
# Sets up the neural network

best_solution = ga.run(target, 5,500)
samples = mcmc.run(target, best_solution, 100, 80)
mcmc.analyse_run(samples)
