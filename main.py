import argparse

import fit_targets as ft
import genetic_algorithm as ga
import mcmc

parser = argparse.ArgumentParser()
parser.add_argument("--nomc",
                    help="ignore mcmc",
                    action="store_true")
args = parser.parse_args()
target = ft.DRTau
if args.nomc:
    print("Here we go again...")
else:
    print("Here goes nothing...")

# Sets up the neural network


best_solution = ga.run(target, 2_500, 25)
if not args.nomc:
    samples = mcmc.run(target, best_solution, 2_000, 50)
    mcmc.analyse_run(samples)
