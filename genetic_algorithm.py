import os

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
import normalisation as norm

import utilities
import variables

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pygad
import pandas as pd
import scipy.stats as stats
import variables as v


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
        flux_object = utilities.JanskyWavelengths(fluxes, wavelengths)
    else:
        flux_object = utilities.SIWavelengths(fluxes, wavelengths)

    # Now, run these fluxes through the neural network.
    # Note that these do not need to be normalised - we will denormalise the NN output instead.
    best_solution = find_solution(flux_object)

    best_fluxes = v.model.predict(np.array([best_solution]), verbose=0)[0]

    best_fluxes = norm.denormalise_fluxes(best_fluxes)

    results = interpolate_fluxes(best_fluxes, flux_object.wavelengths)

    plt.plot(v.wavelengths, best_fluxes, label="Expected values")
    plt.scatter(flux_object.wavelengths, flux_object.convert_to_si(), label="NN values")

    plt.title("Observed SED")
    plt.xlabel("Wavelength (Microns)")
    plt.ylabel("Flux (erg / s / cm^2)")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.show()

    retrieve_inputs(best_solution)


def find_solution(inputs) -> list:
    """
    Runs the genetic algorithm to find a set of solution parameters.
    :return: An array of normalised values that best fit the chi-squared solution.
    """
    print("[GA] Finding solution...")

    gene_spaces = get_gene_spaces()

    # Define our optimisation function to use with the genetic algorithm.
    def optimisor(_ga_instance, free_parameters, _solution_idx):
        # Guess a solution through our neural network.
        results = v.model.predict(np.array([free_parameters]), verbose=0)[0]
        results_denorm = norm.denormalise_fluxes(results)

        # Interpolate the solution over a predetermined number of fluxes.
        results_interp = interpolate_fluxes(results_denorm, inputs.wavelengths)
        chi_squared = stats.chisquare(np.log10(results_interp),
                                      np.log10(np.array(inputs.convert_to_si())),
                                      sum_check=False, ddof=v.split).statistic
        print(f"[{_ga_instance.generations_completed}] {chi_squared}")
        return chi_squared

    # Set up a PyGad instance to apply our chi optimisor to.
    ga = pygad.GA(num_generations=10,
                  num_parents_mating=4,
                  fitness_func=optimisor,
                  sol_per_pop=500,
                  gene_space=gene_spaces,
                  num_genes=variables.split,
                  init_range_low=0.0,
                  init_range_high=1.0,
                  parent_selection_type="tournament",
                  keep_parents=1,
                  crossover_type="single_point",
                  mutation_type="random",
                  mutation_percent_genes=10)
    ga.run()

    # Lastly, return the solution. Could possibly return the other
    # two variables, but doesn't seem necessary (for now)
    sol, sol_fitness, sol_idx = ga.best_solution()
    return sol


def interpolate_fluxes(fluxes, wavelengths):
    """
    Interpolates 100 fluxes from TORUS into `n_interpolate` fluxes using a cubic spline, then returns
    them over a set of predefined wavelengths.
    :param fluxes:
    :param wavelengths:
    :return:
    """
    spline = CubicSpline(v.wavelengths, fluxes, extrapolate=False)
    true_spline = spline(wavelengths)
    return true_spline


def retrieve_inputs(solution):
    n_consts = np.array(pd.read_csv(variables.const_file).transpose())[0]
    i = 0
    for key in v.names.keys():
        if key in variables.excluded: continue
        invert = -1 if v.names[key][1] else 1
        n_solution = solution[i] * invert * n_consts[i]
        if v.names[key][0]:
            result = np.pow(10, n_solution)
            print(f"{key} = {result}")
        else:
            print(f"{key} = {n_solution}")
        i += 1


run([0.545, 0.638, 0.797,
     1.22, 1.63, 2.2,
     3.6, 4.5, 5.8,
     8.0, 24, 61.1,
     70, 74.8, 89.3, 1300], [0.06907, 0.1348, 0.276,
                             0.7187, 0.97, 0.7909,
                             0.5641, 0.4537, 0.4334,
                             0.5358, 1.445, 3.103,
                             3.344, 3.392, 3.048,
                             0.01512], True)
