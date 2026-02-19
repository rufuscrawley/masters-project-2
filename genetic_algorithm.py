import os
from typing import Any

import scipy
from matplotlib import pyplot as plt

import variables

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random
import keras
import pygad
import numpy as np
import pandas as pd
import scipy.stats as stats
from variables import names


def graph_expected(file_name, inputs):
    print("loading model")
    model = keras.models.load_model(f"models/{file_name}_model.keras")

    calculated_flux = model.predict(np.array(inputs), verbose=0)
    plt.plot(variables.wavelengths, calculated_flux[0], label="pred")
    plt.plot([0.545, 0.638, 0.797,
              1.22, 1.63, 2.2,
              3.6, 4.5, 5.8,
              8.0, 24, 61.1,
              70, 74.8, 89.3,
              1300],
             [0.0655, 0.120, 0.216,
              0.483, 0.591, 0.511,
              0.324, 0.220, 0.313,
              0.370, 0.765, 1.42,
              1.581, 1.480, 1.260,
              0.1758], label="im lupi"
             )
    plt.legend()
    plt.show()


def run_genetic_algorithm(inputs, outputs, file_name):
    print("loading model")
    model = keras.models.load_model(f"models/{file_name}_model.keras")
    print("inputs:")
    print(inputs)

    def chi_optimisor(_ga_instance, free_parameters, _solution_idx):
        results = model.predict(np.array([free_parameters]), verbose=0)
        return_value = -1 * stats.chisquare(results[0], np.array(outputs), sum_check=False, ddof=len(inputs)).statistic
        return return_value

    print("setting up ga nistance")
    ga_instance = pygad.GA(num_generations=100,
                           num_parents_mating=4,
                           fitness_func=chi_optimisor,
                           sol_per_pop=8,
                           gene_space=np.arange(0.0, 1.0, 1E-5),
                           num_genes=len(inputs),
                           init_range_low=0.0,
                           init_range_high=1.0,
                           parent_selection_type="sss",
                           keep_parents=1,
                           crossover_type="single_point",
                           mutation_type="random",
                           mutation_percent_genes=33)
    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print(f"Parameters of the best solution : {solution}")
    print(f"Expected Solution: {np.array(inputs)}")
    print(f"Fitness value of the best solution = {solution_fitness}")

    calculated_flux = model.predict(np.array([solution]), verbose=0)
    plt.plot(variables.wavelengths, calculated_flux[0], label="pred")
    plt.plot(variables.wavelengths, outputs, label="exp")
    plt.grid(True)
    plt.legend()
    plt.show()

    # normalisation_constants = np.array(pd.read_csv(f'datasets/constants/{file_name}.csv').transpose())[0]
    # for i, key in enumerate(names.keys()):
    #     mult = -1 if names[key][1] else 1
    #     normalised_solution = solution[i] * mult * normalisation_constants[i]
    #     if names[key][0]:
    #         result = np.pow(10, normalised_solution)
    #         print(f"{key} = 10 ^ {solution[i]} * {mult} * {normalisation_constants[i]} = {result}")
    #     else:
    #         result = normalised_solution
    #         print(f"{key} = {solution[i]} * {mult} * {normalisation_constants[i]} = {result}")


def get_dataset_from_csv(dataset_file, data_split) -> tuple[Any, Any]:
    random.seed()
    dataset = pd.read_csv(f"datasets/normalised/{dataset_file}_norm.csv")
    i, o = dataset.iloc[:, :data_split], dataset.iloc[:, data_split:]
    ROW = random.randint(1, 50_000)
    # ROW = 2
    return np.array(i.iloc[ROW]), np.array(o.iloc[ROW])


def run():
    inputs, outputs = get_dataset_from_csv("outputs", 14)
    run_genetic_algorithm(inputs, outputs, "outputs")
