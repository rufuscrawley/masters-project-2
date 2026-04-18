import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from matplotlib import pyplot as plt
import scipy.stats as stats
import pandas as pd
import numpy as np
import pygad
import variables as v
import network_variables as nv
import utilities


def get_gene_spaces():
    # Set up the gene space for our parameters.
    split_value = 0
    n_csv = pd.read_csv(v.n_file)
    gene_spaces = []
    for column in n_csv:
        if split_value == v.split:
            break
        gene_spaces.append({"low": n_csv[column].min(),
                            "high": n_csv[column].max()})
        split_value += 1
    return gene_spaces


def run(parameters,
        generations=5, sol_per_pop=1000):
    # Set up our objects with the fluxes and wavelengths attached.
    wavelengths, fluxes = parameters
    # Now, run these fluxes through the neural network.
    # Note that these do not need to be normalised - we will denormalise the NN output instead.
    best_solution = find_solution(wavelengths, fluxes,
                                  generations, sol_per_pop)
    print("Found solution, plotting SED...")
    # Plot our GA solution
    best_fluxes = nv.predict_fluxes(best_solution, True)
    plt.plot(v.wavelengths, best_fluxes, label="NN (predicted)")

    # Lastly, plot the interpolated input values
    plt.scatter(wavelengths, fluxes, label="Fluxes (raw)")

    plt.title("Observed SED")
    plt.xlabel("Wavelength (Microns)")
    plt.ylabel("Flux (erg / s / cm^2)")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.ylim(10e-15, None)
    plt.show()

    return best_solution


def find_solution(wavelengths, fluxes, generations, sol_per_pop) -> list:
    """
    Runs the genetic algorithm to find a set of solution parameters.
    :return: An array of normalised values that best fit the chi-squared solution.
    """

    # Define our optimisation function to use with the genetic algorithm.
    def optimisor(_ga_instance, free_parameters, _solution_idx):
        # Suggest a set of free parameters, and then use NN to predict
        # a denormalised set of 100 fluxes.
        sol_guessed = nv.predict_fluxes(free_parameters, True)
        # Interpolate the solution over a predetermined number of fluxes.
        sol_interp = utilities.interpolate_fluxes(sol_guessed, wavelengths)
        # Find chi-squared value - taking negative as PyGad optimises for minimum
        chi_squared = -1 * stats.chisquare(sol_interp, (np.array(fluxes)),
                                           sum_check=False, ddof=v.split).statistic
        return chi_squared

    def on_generation(ga_instance):
        generation_num = ga_instance.generations_completed
        best_fitness = ga_instance.best_solutions_fitness if ga_instance.best_solutions_fitness else 0
        print(f"Generation {generation_num} reached. Best Fitness: {best_fitness[-1]} ")

    # Set up a PyGad instance to apply our chi optimisor to.
    ga = pygad.GA(num_generations=generations,
                  num_parents_mating=int(sol_per_pop / 10),
                  fitness_func=optimisor,
                  sol_per_pop=sol_per_pop,
                  gene_space=get_gene_spaces(),
                  num_genes=v.split,
                  init_range_low=0.0,
                  init_range_high=1.0,
                  parent_selection_type="random",
                  keep_parents=int(sol_per_pop / 40),
                  crossover_type="single_point",
                  mutation_type="random",
                  mutation_percent_genes=20,
                  on_generation=on_generation)
    ga.run()
    ga.plot_fitness()
    sol, sol_fitness, sol_idx = ga.best_solution()
    print(f"fitness: {sol_fitness}")
    return sol
