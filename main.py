import argparse

import fit_targets as ft
import genetic_algorithm as ga
import mcmc

parser = argparse.ArgumentParser()
parser.add_argument("--nomc",
                    help="ignore mcmc",
                    action="store_true")
args = parser.parse_args()
target = ft.FTTau

if args.nomc:
    print("Here we go again...")
else:
    print("Here goes nothing...")

# Sets up the neural network


best_solution = ga.run(target, 2500, 50)
if not args.nomc:
    samples = mcmc.run(target, best_solution, 3_500, 75)
    mcmc.analyse_run(samples)
