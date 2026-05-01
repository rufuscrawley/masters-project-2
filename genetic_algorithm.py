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
    n_csv = pd.read_csv(ve.n_file)
    gene_spaces = []
    for variable in ve.included:
        if ve.included[variable] is not None:
            gene_spaces.append(ve.included[variable])
            continue
        gene_spaces.append({"low": n_csv[variable].min() * 1.01,
                            "high": n_csv[variable].max() * 0.99})

    # Now apply gene space for extinction (our expected final value)
    gene_spaces.append({"low": -3.0,
                        "high": 3.0})

    return gene_spaces


def run(target: FitObject, generations=5, sol_per_pop=1000):
    # Run these fluxes through the neural network.
    # Note that these do not need to be normalised - we will denormalise the NN output instead.
    best_solution, fitness = find_solution(target, generations, sol_per_pop)
    print("Found solution, plotting SED...")

    best_fluxes = vl.predict_fluxes(best_solution[:-1], True)
    vl.apply_extinction(best_fluxes, best_solution[-1])

    # Lastly, plot the interpolated input values
    plt.plot(ve.wavelengths, best_fluxes,
             label=f"NN (predicted)", color="k")
    plt.scatter(target.wavelengths, target.fluxes,
                label="Fluxes (raw)", marker="x")
    plt.errorbar(target.wavelengths, target.fluxes,
                 yerr=target.flux_err, fmt='none')
    plt.title(f"Observed SED of {target.name} ($\chi^2$ ~ {-1 * np.round(fitness, 4)})")
    plt.xlabel("Wavelength ($\mu$m)")
    plt.ylabel("$\lambda$F (erg / s / cm$^2$)")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.xlim(np.min(target.wavelengths) / 2, np.max(target.wavelengths) * 2)
    plt.ylim(np.min(target.fluxes) / 2, np.max(target.fluxes) * 2)
    plt.tight_layout()
    plt.show()

    # Alternately, test the others
    fig, axs = plt.subplots(2, 1)
    sol_interp = vl.interpolate_fluxes(best_fluxes, target.wavelengths)
    fig.suptitle(f"Interpolated SED of {target.name}")

    axs[0].scatter(target.wavelengths, sol_interp, label="Predicted flux", color="k", marker="x")
    axs[0].scatter(target.wavelengths, target.fluxes, label="Expected flux", marker="x")
    axs[0].set_ylabel("$\lambda$F (erg / s / cm$^2$)")
    axs[0].set_xscale("log")
    axs[0].set_yscale("log")
    axs[0].legend()

    deviations = (np.log10(target.fluxes) - np.log10(sol_interp)) * 100 / np.log10(target.fluxes)
    axs[1].plot(target.wavelengths, deviations,
                label=f"Predicted flux ($\mu$ = {np.round(np.mean(deviations), 3)})",
                color="k")
    axs[1].axhline(0.0)
    axs[1].set_ylabel("% deviation from expected F")
    axs[1].set_xscale("log")

    plt.xlabel("Wavelength ($\mu$m)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return best_solution


def find_solution(target: FitObject, generations, sol_per_pop) -> list:
    """
    Runs the genetic algorithm to find a set of solution parameters.
    :return: An array of normalised values that best fit the chi-squared solution.
    """

    ddof = len(ve.wavelengths) - (len(ve.included) - 1)
    flux_exp = np.log10(target.fluxes)
    err = target.flux_err / (target.fluxes * np.log(10))

    # Define our optimisation function to use with the genetic algorithm.
    def optimisor(_ga_instance, free_parameters, _solution_idx):
        # Suggest a set of free parameters, and then use NN to predict
        # a denormalised set of 100 fluxes
        flux_guess = vl.predict_fluxes(free_parameters[:-1], True)
        # Apply extinction to the fluxes
        vl.apply_extinction(flux_guess, free_parameters[-1])
        # Interpolate the solution over a predetermined number of fluxes
        flux_interp = vl.interpolate_fluxes(flux_guess, target.wavelengths)
        flux_obs = np.log10(flux_interp)
        mse = chisquare(flux_obs / err, flux_exp / err, sum_check=False, ddof=ddof).statistic
        return mse

    def on_generation(ga_instance):
        generation_num = ga_instance.generations_completed
        best_fitness = ga_instance.best_solutions_fitness if ga_instance.best_solutions_fitness else 0
        print(f"({generation_num} / {generations}) - {-1 * best_fitness[-1]}")

    # Set up a PyGad instance to apply our chi optimisor to.
    ga = pygad.GA(num_generations=generations,
                  num_parents_mating=int(sol_per_pop / 4),
                  fitness_func=optimisor,
                  sol_per_pop=sol_per_pop,
                  gene_space=get_gene_spaces(),
                  num_genes=(ve.split + 1),
                  init_range_low=-1.0,
                  init_range_high=1.0,
                  parent_selection_type="random",
                  keep_parents=int(sol_per_pop / 8),
                  crossover_type="single_point",
                  mutation_type="random",
                  mutation_percent_genes=33,
                  on_generation=on_generation)
    print("Running PyGad...")
    ga.run()
    ga.plot_fitness()
    sol, sol_fitness, sol_idx = ga.best_solution()
    print(f"Final chi-squared: {-1 * sol_fitness}")
    return sol, sol_fitness
