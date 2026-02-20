import os

from matplotlib import pyplot as plt

import utilities
import variables

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras
import pygad
import numpy as np
import pandas as pd
import scipy.stats as stats
from variables import names


def get_gene_space(file_path, n_parameters):
    normie_x = 0
    n_file = f'datasets/normalised/n_{file_path}.csv'
    n_csv = pd.read_csv(n_file)
    space_list = []
    for column in n_csv:
        if normie_x == n_parameters:
            print(f"{normie_x} = {n_parameters}")
            break
        space_list.append({"low": n_csv[column].min(),
                           "high": n_csv[column].max()})
        normie_x += 1
    return space_list


def find_solution(outputs, file_name, input_length, inputs=None):
    model = keras.models.load_model(f"models/{file_name}_model.keras")

    def chi_optimisor(_ga_instance, free_parameters, _solution_idx):
        results = model.predict(np.array([free_parameters]), verbose=0)
        return_value = -1 * stats.chisquare(results[0], np.array(outputs),
                                            sum_check=False, ddof=input_length).statistic
        gens = ga_instance.generations_completed
        print(f"gen: {gens} || chi: {return_value * -1}")
        return return_value

    print("setting up ga nistance")
    ga_instance = pygad.GA(num_generations=1,
                           num_parents_mating=8,
                           fitness_func=chi_optimisor,
                           sol_per_pop=16,
                           gene_space=get_gene_space("outputs", 13),
                           num_genes=input_length,
                           init_range_low=0.0,
                           init_range_high=1.0,
                           parent_selection_type="sss",
                           keep_parents=4,
                           crossover_type="single_point",
                           mutation_type="random",
                           mutation_percent_genes=10)
    ga_instance.run()
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print(f"PARAMS : {solution}")
    if inputs.any():
        print(f"EXPECTED : {np.array(inputs)}")
    print(f"FITNESS : {solution_fitness}")

    return solution


def retrieve_inputs(solution, file_name, ignored_variables):
    file = f'../datasets/constants/const_{file_name}.csv'
    n_consts = np.array(pd.read_csv(file).transpose())[0]
    i = 0

    for key in names.keys():
        if key in ignored_variables: continue
        mult = -1 if names[key][1] else 1
        n_solution = solution[i] * mult * n_consts[i]
        if names[key][0]:
            result = np.pow(10, n_solution)
            print(f"{key} = {result}")
        else:
            result = n_solution
            print(f"{key} = {result}")
        i += 1


def graph_outputs(solution, file_name, outputs_csv):
    model = keras.models.load_model(f"models/{file_name}_model.keras")
    outputs = model.predict(np.array([solution]), verbose=0)[0]
    n_consts = utilities.get_y_consts(file_name, len(solution))

    for n, const in enumerate(n_consts):
        og_output = outputs[n]
        if og_output == 0.0:
            outputs[n] = outputs[n - 1]
            continue
        outputs[n] = np.pow(10, og_output * -1 * const)

    plt.plot(variables.wavelengths, outputs, label="plotted")
    plt.plot(variables.wavelengths, outputs_csv, label="expected")

    plt.title("Observed SED")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Wavelength (Microns)")
    plt.ylabel("Flux (erg / s / cm^2)")
    plt.legend()
    plt.show()


def run():
    inputs, outputs, n_inputs, n_outputs = utilities.get_dataset_from_csv("outputs", 13)
    sol = find_solution(n_outputs, "outputs", len(n_inputs), n_inputs)
    graph_outputs(sol, "outputs", outputs)
    # retrieve_inputs(sol, "outputs", ["alphadisc"])
