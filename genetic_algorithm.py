import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from matplotlib import pyplot as plt
import scipy.stats as stats
import pandas as pd
import numpy as np
import pygad

import normalisation as norm
import variables as v
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


def run(wavelengths, fluxes, janskys=False):
    # Set up our objects with the fluxes and wavelengths attached.

    if janskys:
        fluxes = utilities.JanskyWavelengths(fluxes, wavelengths).convert_to_si()
    # Now, run these fluxes through the neural network.
    # Note that these do not need to be normalised - we will denormalise the NN output instead.
    best_solution = find_solution(fluxes, wavelengths)

    # Plot our GA solution
    best_fluxes = v.model.predict(np.array([best_solution]), verbose=0)[0]
    best_fluxes = norm.denormalise_fluxes(best_fluxes)
    plt.plot(v.wavelengths, best_fluxes, label="NN (predicted)")

    # For reference, plot what our NN expects
    # exp_solution = [
    #     0.014783015,
    #     936.3319832,
    #     44.25,
    #     96.41907064,
    #     5.622734881,
    #     1.35035425,
    #     0.002611227,
    #     2.211162497,
    #     8883.089312
    # ]

    # exp_solution = [
    #
    # ]
    # exp_solution = norm.normalise_inputs(exp_solution)
    # exp_fluxes = v.model.predict(np.array([exp_solution]), verbose=0)[0]
    # exp_fluxes = norm.denormalise_fluxes(exp_fluxes)
    # plt.plot(v.wavelengths, exp_fluxes, label="NN (expected)")

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

    norm.denormalise_inputs(best_solution)


def find_solution(fluxes, wavelengths) -> list:
    """
    Runs the genetic algorithm to find a set of solution parameters.
    :return: An array of normalised values that best fit the chi-squared solution.
    """

    # Define our optimisation function to use with the genetic algorithm.
    def optimisor(_ga_instance, free_parameters, _solution_idx):
        # Suggest a set of free parameters, and then use NN to predict
        # a denormalised set of 100 fluxes.
        sol_guessed = v.model.predict(np.array([free_parameters]), verbose=0)[0]
        s_1 = norm.denormalise_fluxes(sol_guessed)
        # Interpolate the solution over a predetermined number of fluxes.
        sol_interp = norm.interpolate_fluxes(s_1, wavelengths)
        # Find chi-squared value - taking negative as PyGad optimises for minimum
        chi_squared = -1 * stats.chisquare(sol_interp, (np.array(fluxes)),
                                           sum_check=False, ddof=v.split).statistic
        print(f"[{_ga_instance.generations_completed}] {chi_squared}")
        return chi_squared

    # Set up a PyGad instance to apply our chi optimisor to.
    ga = pygad.GA(num_generations=16,
                  num_parents_mating=8,
                  fitness_func=optimisor,
                  sol_per_pop=512,
                  gene_space=get_gene_spaces(),
                  num_genes=v.split,
                  init_range_low=0.0,
                  init_range_high=1.0,
                  parent_selection_type="random",
                  keep_parents=2,
                  crossover_type="single_point",
                  mutation_type="random",
                  mutation_percent_genes=15)
    ga.run()
    ga.plot_fitness()
    # Lastly, return the solution. Could possibly return the other
    # two variables, but doesn't seem necessary (for now)
    sol, sol_fitness, sol_idx = ga.best_solution()
    print(f"fitness: {sol_fitness}")
    return sol


run([0.545, 0.638, 0.797, 1.22, 1.63, 2.2,
     3.6, 4.5, 5.8, 8.0, 24, 61.1,
     70, 74.8, 89.3, 1300],
    [0.06907, 0.1348, 0.276, 0.7187, 0.97, 0.7909,
     0.5641, 0.4537, 0.4334, 0.5358, 1.445, 3.103,
     3.344, 3.392, 3.048, 0.01512], True)
