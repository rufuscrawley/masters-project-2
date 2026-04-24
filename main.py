# Central file to collate all relevant functions

import fit_targets as ft
import genetic_algorithm as ga
import mcmc
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--nomc",
                    help="ignore mcmc",
                    action="store_true")
args = parser.parse_args()
target = ft.IMLupi
if args.nomc:
    print("Here we go again...")
else:
    print("Here goes nothing...")

# Sets up the neural network

best_solution = ga.run(target, 10, 200)
if not args.nomc:
    samples = mcmc.run(target, best_solution, 1_500, 25)
    mcmc.analyse_run(samples)
