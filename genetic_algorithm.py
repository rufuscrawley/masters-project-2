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


def find_solution(inputs, outputs, file_name):
    model = keras.models.load_model(f"models/{file_name}_model.keras")

    def chi_optimisor(_ga_instance, free_parameters, _solution_idx):
        results = model.predict(np.array([free_parameters]), verbose=0)
        return_value = -1 * stats.chisquare(results[0], np.array(outputs),
                                            sum_check=False, ddof=len(inputs)).statistic
        gens = ga_instance.generations_completed
        print(f"gen: {gens} || chi: {return_value}")
        return return_value

    print("setting up ga nistance")
    ga_instance = pygad.GA(num_generations=100,
                           num_parents_mating=8,
                           fitness_func=chi_optimisor,
                           sol_per_pop=16,
                           gene_space={"low": 0, "high": 1},
                           num_genes=len(inputs),
                           init_range_low=0.0,
                           init_range_high=1.0,
                           parent_selection_type="tournament",
                           K_tournament=4,
                           keep_parents=2,
                           crossover_type="single_point",
                           mutation_type="random",
                           mutation_percent_genes=10)
    ga_instance.run()
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print(f"Parameters of the best solution : {solution}")
    print(f"Expected Solution: {np.array(inputs)}")
    print(f"Fitness value of the best solution = {solution_fitness}")

    return solution


def retrieve_inputs(solution, file_name, ignored_variables):
    file = f'datasets/constants/const_{file_name}.csv'
    n_consts = np.array(pd.read_csv(file).transpose())[0]
    i = 0

    for key in names.keys():
        if key in ignored_variables: continue
        mult = -1 if names[key][1] else 1
        n_solution = solution[i] * mult * n_consts[i]
        if names[key][0]:
            result = np.pow(10, n_solution)
            print(f"{key} = 10 ^ {solution[i]} * {mult} * {n_consts[i]} = {result}")
        else:
            result = n_solution
            print(f"{key} = {solution[i]} * {mult} * {n_consts[i]} = {result}")
        i += 1


def graph_outputs(solution, file_name, outputs_csv):
    model = keras.models.load_model(f"models/{file_name}_model.keras")
    outputs = model.predict(np.array([solution]), verbose=0)[0]
    file = f'datasets/constants/const_{file_name}.csv'
    n_consts = np.array(pd.read_csv(file).transpose())[0].tolist()
    n_consts = n_consts[len(solution):]
    for n, const in enumerate(n_consts):
        og_output = outputs[n]
        if og_output == 0.0:
            outputs[n] = outputs[n - 1]
            continue
        outputs[n] = (og_output * -1 * const)

    plt.plot(np.log10(variables.wavelengths), outputs, label="plotted")
    plt.plot(np.log10(variables.wavelengths), np.log10(outputs_csv))

    plt.legend()
    plt.show()


def run():
    inputs, outputs, n_inputs, n_outputs = utilities.get_dataset_from_csv("outputs", 13)
    sol = find_solution(n_inputs, n_outputs, "outputs")
    graph_outputs(sol, "outputs", outputs)
    # retrieve_inputs(sol, "outputs", ["alphadisc"])
