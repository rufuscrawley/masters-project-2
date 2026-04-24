import os

from scipy.stats import chisquare

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import pygad
import variables_early as ve
import variables_late as vl

from fit_targets import FitObject


def get_gene_spaces():
    # Set up the gene space for our parameters.
    split_value = 0
    n_csv = pd.read_csv(ve.n_file)

    gene_spaces = []
    for column in n_csv:
        if split_value == ve.split:
            break
        gene_spaces.append({"low": n_csv[column].min() * 1.05,
                            "high": n_csv[column].max() * 0.95})
        split_value += 1

    # Now apply gene space for extinction (our expected final value)
    gene_spaces.append({"low": 0.0,
                        "high": 3.0})

    return gene_spaces


def run(target: FitObject, generations=5, sol_per_pop=1000):
    # Run these fluxes through the neural network.
    # Note that these do not need to be normalised - we will denormalise the NN output instead.
    best_solution = find_solution(target, generations, sol_per_pop)
    print("Found solution, plotting SED...")

    best_fluxes = vl.predict_fluxes(best_solution[:-1], True)
    vl.apply_extinction(best_fluxes, best_solution[-1])

    # Lastly, plot the interpolated input values
    plt.plot(ve.wavelengths, best_fluxes, label=f"NN (predicted) (Ext = {best_solution[-1]})")
    plt.scatter(target.wavelengths, target.fluxes, label="Fluxes (raw)", linewidths=.1)
    plt.errorbar(target.wavelengths, target.fluxes, yerr=target.flux_err, fmt='none')
    plt.title(f"Observed SED of {target.name}")
    plt.xlabel("Wavelength (Microns)")
    plt.ylabel("Flux (erg / s / cm^2)")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.ylim(10e-15, None)
    plt.show()

    # Alternately, test the others
    sol_interp = vl.interpolate_fluxes(best_fluxes, target.wavelengths)
    plt.scatter(target.wavelengths, sol_interp, linewidths=.05, label="Predicted")
    plt.scatter(target.wavelengths, target.fluxes, linewidths=.05, label="Expected")
    plt.title(f"Interpolated SED of {target.name}")
    plt.xlabel("Wavelength (Microns)")
    plt.ylabel("Flux (erg / s / cm^2)")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.ylim(10e-15, None)

    plt.show()

    plt.show()

    return best_solution


def find_solution(target: FitObject, generations, sol_per_pop) -> list:
    """
    Runs the genetic algorithm to find a set of solution parameters.
    :return: An array of normalised values that best fit the chi-squared solution.
    """

    # Define our optimisation function to use with the genetic algorithm.
    def optimisor(_ga_instance, free_parameters, _solution_idx):
        # Suggest a set of free parameters, and then use NN to predict
        # a denormalised set of 100 fluxes.
        sol_guessed = vl.predict_fluxes(free_parameters[:-1], True)

        # Apply extinction to the fluxes
        vl.apply_extinction(sol_guessed, free_parameters[-1])

        # Interpolate the solution over a predetermined number of fluxes.
        sol_interp = vl.interpolate_fluxes(sol_guessed, target.wavelengths)

        sol_interp = np.log10(sol_interp)
        l_fluxes = np.log10(target.fluxes)

        mse = chisquare(sol_interp, l_fluxes, sum_check=False).statistic
        return mse

    def on_generation(ga_instance):
        generation_num = ga_instance.generations_completed
        best_fitness = ga_instance.best_solutions_fitness if ga_instance.best_solutions_fitness else 0
        print(f"Generation {generation_num} finished. Best Fitness: {-1 * best_fitness[-1]} ")

    # Set up a PyGad instance to apply our chi optimisor to.
    ga = pygad.GA(num_generations=generations,
                  num_parents_mating=int(sol_per_pop / 10),
                  fitness_func=optimisor,
                  sol_per_pop=sol_per_pop,
                  gene_space=get_gene_spaces(),
                  num_genes=(ve.split + 1),
                  init_range_low=0.25,
                  init_range_high=0.75,
                  parent_selection_type="random",
                  keep_parents=int(sol_per_pop / 40),
                  crossover_type="single_point",
                  mutation_type="random",
                  mutation_percent_genes=33,
                  on_generation=on_generation)
    ga.run()
    ga.plot_fitness()
    sol, sol_fitness, sol_idx = ga.best_solution()
    print(f"fitness: {sol_fitness}")
    return sol
