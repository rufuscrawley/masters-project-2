import random
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras
import pygad
import numpy as np

import chi_squared
import pandas as pd

from variables import names

random.seed()
ROW = random.randint(1, 50_000)

data = pd.read_csv('datasets/normalised.csv')
reconstructed_model = keras.models.load_model("models/final_model.keras")

x, y = data.iloc[:, :14], data.iloc[:, 14:]
x_row, y_row = x.iloc[ROW], y.iloc[ROW]


def chi_optimisor(ga_instance, free_parameters, solution_idx):
    results = reconstructed_model.predict(np.array([free_parameters]), verbose=0)
    return_value = chi_squared.reduced_chi_square(results[0], np.array(y.iloc[ROW]))
    print(return_value)
    return -return_value


ga_instance = pygad.GA(num_generations=1000,
                       num_parents_mating=4,
                       fitness_func=chi_optimisor,
                       sol_per_pop=8,
                       num_genes=len(x_row),
                       init_range_low=0,
                       init_range_high=1,
                       parent_selection_type="sss",
                       keep_parents=1,
                       crossover_type="single_point",
                       mutation_type="random",
                       mutation_percent_genes=10)

ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
