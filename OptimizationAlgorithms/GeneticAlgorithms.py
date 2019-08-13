import random as rd
import numpy as np
import math
import copy
import time

#NB: These functions are not secure, anyone with access to them will have access to the functions permission level.
#If you are concerned about security do an analysis of all the code before using.

MUTATION_SIGMA = .3

#Some Static Functions

"""
Function: random_population
---------------------------
Computes a random population of size (count) with axes specified.

count: integer amount of items in population
axes: list or tuple of length 2 lists or tuples. Each of these 
tuples/lists define the lower and upper bound on their respective
axes ex: [(0, 10), (0, 5)] would mean that there are two axes, on 
from 0 to 10 and the other from 0 to 5

returns: A random population from the input specifications
"""
def random_population(count, axes):
    population = np.empty((count, len(axes)))
    for i in range(count):
        for j in range(len(axes)):
            population[i][j] = rd.uniform(axes[j][0], axes[j][1])
    return population

"""
Function: analogue_random_population
------------------------------------
Computes a random population of size (count) with axes specified. Will only pick values inside the axes, the not range 
of the axes. This function should be used for non-continuous functions

count: integer amount of items in population
axes: list or tuple of indiscriminate length lists or tuples. Each of
these describe all inputs allowed along said axis
axes ex: TBC

returns: A random population from the input specifications
"""
def analogue_random_population(count, axes):
    population = np.empty((count, len(axes)))
    for i in range(count):
        for j in range(len(axes)):
            population[i][j] = rd.choice(axes[j])
    return population

"""
Function: mutate
----------------
Computes a mutation each member of a population.

population: population to be mutated
axes: axes of the population
generations: number of generations into genetic algorithm
MUTATION_CHANCE: chance at a mutation happening
MUTATION_DECREASE_PER_GENERATION: decreased chance of mutation per generation

returns: a population that has been randomly mutated that still lies within the axes specified.
"""
def mutate(population, axes, generations=0,
           MUTATION_CHANCE=.3, MUTATION_DECREASE_PER_GENERATION=0):
    newdna = np.empty(population.shape)
    for i in range(population.shape[0]):
        for j in range(population.shape[1]):
            c = population[i][j]
            if rd.random() < MUTATION_CHANCE - MUTATION_DECREASE_PER_GENERATION * generations:
                c = rd.uniform(axes[j][0], axes[j][1])
            newdna[i][j] = c
    return newdna

"""
Function: analogue_mutate
-------------------------
Computes a mutation each member of a non-continuous population.

population: population to be mutated
axes: axes of the population
generations: number of generations into genetic algorithm
MUTATION_CHANCE: chance at a mutation happening
MUTATION_DECREASE_PER_GENERATION: decreased chance of mutation per generation

returns: a population that has been randomly mutated that still lies within the axes specified.
"""
def analogue_mutate(population, axes, generations=0,
           MUTATION_CHANCE=.3, MUTATION_DECREASE_PER_GENERATION=0):
    newdna = np.empty(population.shape)
    for i in range(population.shape[0]):
        for j in range(population.shape[1]):
            c = population[i][j]
            if rd.random() < MUTATION_CHANCE - MUTATION_DECREASE_PER_GENERATION * generations:
                c = rd.choice(axes[j])
            newdna[i][j] = c
    return newdna

"""
Function: my_gauss
------------------
TBC
"""
def my_gauss(val, axis):
    newval = rd.gauss(val, MUTATION_SIGMA)
    return max(min(newval, axis[1]), axis[0])

"""
Function: functional_mutation
-----------------------------
Mutates a population based on a function

population: population to be mutated
axes: axes of the population
function: Individual -> axes -> Individual

returns: A population that has been mutated based on the function given.
"""
def functional_mutation(population, axes, function=my_gauss):
    newdna = np.empty(population.shape)
    for i in range(population.shape[0]):
        for j in range(population.shape[1]):
            c = population[i][j]
            axis = axes[j]
            c = function(c, axis)
            newdna[i][j] = c
    return newdna

"""
Function: cross
---------------
Calculates a genetic crossover between two strands

a: first dna sequence
b: second dna sequence

returns: their (random) cross
"""
def cross(a, b, CROSSOVER_CHANCE=.3):
    c = np.empty((a.shape[0]))
    for i in range(a.shape[0]):
        if rd.random() < CROSSOVER_CHANCE:
            if rd.random() > .5:
                c[i] = a[i]
            else:
                c[i] = b[i]
        else:
            c[i] = (a[i] + b[i]) / 2
    return c

"""
Function: analogue_cross
------------------------
Calculates a genetic crossover between two strands. Non-continuous version, doesn't mix strains.

a: first dna sequence
b: second dna sequence

returns: their (random) cross
"""
def analogue_cross(a, b):
    c = np.empty((a.shape[0]))
    for i in range(a.shape[0]):
        if rd.random() > .5:
            c[i] = a[i]
        else:
            c[i] = b[i]
    return c

"""
Function: gaussiancross
-----------------------
Calculates a genetic crossover between two strands

dna1: first dna sequence
dna2: second dna sequence
axes: omit

returns: their (gaussian) cross
"""
def gaussiancross(dna1, dna2, axes = None):
    c = np.empty(dna1.shape[0])
    for i in range(dna1.shape[0]):
        c[i] = max(min(rd.gauss((dna1[i] + dna2[i]) / 2, abs(dna1[i] - dna2[i]) / 5),
                       max(dna1[i], dna2[i])),
                   min(dna1[i], dna2[i]))
    return c

"""
Function: nearestpoint
----------------------
Calculates the nearest of a continuous number along a non-continuous axis

point: continuous point
axis: axis along which the nearest should be found

returns: point along axis nearest to "point"
"""
def nearestpoint(point, axis):
    return axis[np.argmin(abs(np.array(axis)-point))]

"""
Function: analogue_gaussiancross
--------------------------------
Calculates a genetic crossover between two strands

dna1: first dna sequence
dna2: second dna sequence
axes: axes of the dna

returns: their (gaussian) cross on a non-continuous (shape?)
"""
def analogue_gaussiancross(dna1, dna2, axes=None):
    c = gaussiancross(dna1, dna2)
    for i in range(len(c)):
        c[i] = nearestpoint(c[i], axes[i])
    return c

"""
Class: PopulationAlgorithm
--------------------------
Class for use of Population Genetic Algorithm

SECURITY WARNING: THIS IS NOT A SECURE IMPLEMENTATION, DON'T GIVE THIS FUNCTION ANY PERMISSIONS YOU DON'T WANT END USER TO HAVE
"""
class PopulationAlgorithm:
    locals = {
        "fitness_function": None,  # Function to be optimized: POINT -> POPULATION -> VALUE
        "population_size": -1,  # How large should the population be
        "time_constraint": 120,  # How long should the algorithm run
        "axes": None,  # What are the axes of the algorithm
        "population_function": random_population,  # What function should be used to create a random population
        "mutate_function": mutate,  # The mutation function
        "cross_function": cross,  # The crossover function
        "weighting_bias": lambda x: x,  # Bias function for weighting
    }

    """
    Function: PopulationAlgorithm.__init__
    -----------------------------------
    Initializes a GeneticAlgorithm class instance
    
    params: dictionary of parameters to set
    """
    def __init__(self, params={}):
        failure = False
        for param, value in params.items():
            try:
                self.locals[param] = value  # Security Risk
            except:
                failure = True
        if failure:
            print("Failure in input parameters")  # Could move this into loop and be more specific?

    """
    Function: PopulationAlgorithm.set
    ------------------------------
    Sets the value of a parameter

    param: String of parameter to set
    value: String of value for parameter to be set to. NB: to make phi = "lambda". value must be ""lambda""

    returns 0 upon failure, 1 if succeeded
    """
    def set(self, param, value):
        try:
            self.locals[param] = value  # Security Risk
        except:
            return 0
        return 1

    """
    Function: PopulationAlgorithm.cross
    -----------------------------------
    Completes cross on entire space based on weightings

    weighted_population: population with weighting summing to 1
    run_locals: local run by run information
    
    returns: the new population made from crossing
    """

    def cross(self, weighted_population, run_locals):
        new_population = np.empty((weighted_population.shape[0], weighted_population.shape[1] - 1))
        for i in range(weighted_population.shape[0]):
            new_population[i] = run_locals["cross_function"](weighted_population[:, 1:][np.random.choice(
                                                                 np.arange(weighted_population.shape[0]),
                                                                 p=weighted_population[:, 0])],
                                                             weighted_population[:, 1:][np.random.choice(
                                                                 np.arange(weighted_population.shape[0]),
                                                                 p=weighted_population[:, 0])])
        return new_population

    """
    Functions: PopulationAlgorithm.run
    ----------------------------------
    Runs the genetic algorithm.

    temp_params: parameters set for this run only

    returns: final population
    """
    def run(self, temp_params={}):
        run_locals = copy.deepcopy(self.locals)
        for param, value in temp_params.items():
            run_locals[param] = value
        population = run_locals["population_function"](run_locals["population_size"], run_locals["axes"])
        start_time = time.time()
        while time.time() < start_time + run_locals["time_constraint"]:
            weightings = np.apply_along_axis(run_locals["fitness_function"], 1, population, population).reshape((population.shape[0], 1))
            weightings[weightings < 0] = 0
            weightings = weightings / np.sum(weightings)
            weighted_population = np.concatenate((weightings, population), axis=1)
            population = functional_mutation(self.cross(weighted_population, run_locals), run_locals["axes"])
        return population

"""
Class: GeneticAlgorithm
-----------------------
Class for use of Genetic Algorithm

SECURITY WARNING: THIS IS NOT A SECURE IMPLEMENTATION, DON'T GIVE THIS FUNCTION ANY PERMISSIONS YOU DON'T WANT END USER TO HAVE
"""
class GeneticAlgorithm:
    locals = {
        "fitness_function": None,  # Function to be optimized: POINT -> VALUE
        "population_size": -1,  # How large should the population be
        "time_constraint": 120,  # How long should the algorithm run
        "axes": None,  # What are the axes of the algorithm
        "population_function": random_population,  # What function should be used to create a random population
        "mutate_function": mutate,  # The mutation function
        "cross_function": gaussiancross,  # The crossover function
        "weighting_bias": lambda x: x,  # Bias function for weighting... NOT IMPLEMENTED YET
    }

    """
    Function: GeneticAlgorithm.__init__
    -----------------------------------
    Initializes a GeneticAlgorithm class instance
    
    params: dictionary of parameters to set
    """
    def __init__(self, params={}):
        failure = False
        for param, value in params.items():
            try:
                self.locals[param] = value #Security Risk
            except:
                failure = True
        if failure:
            print("Failure in input parameters") #Could move this into loop and be more specific?

    """
    Function: GeneticAlgorithm.set
    ------------------------------
    Sets the value of a parameter
    
    param: String of parameter to set
    value: String of value for parameter to be set to. NB: to make phi = "lambda". value must be ""lambda""
    
    returns 0 upon failure, 1 if succeeded
    """
    def set(self, param, value):
        try:
            self.locals[param] = value #Security Risk
        except:
            return 0
        return 1

    """
    Function: GeneticAlgorithm.cross
    --------------------------------
    Completes cross on entire space based on weightings
    
    weighted_population: population with weighting summing to 1
    run_locals: local run by run information
    
    """
    def cross(self, weighted_population, run_locals):
        new_population = np.empty((weighted_population.shape[0], weighted_population.shape[1]-1))
        for i in range(weighted_population.shape[0]):
            new_population[i] = run_locals["cross_function"](weighted_population[:, 1:][np.random.choice(
                                                                 np.arange(weighted_population.shape[0]),
                                                                 p=weighted_population[:, 0])],
                                                             weighted_population[:, 1:][np.random.choice(
                                                                 np.arange(weighted_population.shape[0]),
                                                                 p=weighted_population[:, 0])],
                                                             run_locals["axes"])
        return new_population

    """
    Functions: GeneticAlgorithm.run
    -------------------------------
    Runs the genetic algorithm.
    
    temp_params: parameters set for this run only
    
    returns: choice with maximum value found across all generations
    """
    def run(self, temp_params={}):
        run_locals = copy.deepcopy(self.locals)
        for param, value in temp_params.items():
            run_locals[param] = value
        population = run_locals["population_function"](run_locals["population_size"], run_locals["axes"])
        overall_max = None
        overall_max_val = 0
        start_time = time.time()
        while time.time() < start_time + run_locals["time_constraint"]:
            weightings = np.apply_along_axis(run_locals["fitness_function"], 1, population).reshape((population.shape[0], 1))
            weightings[weightings < 0] = 0
            if overall_max_val < weightings.max():
                overall_max_val = weightings.max()
                overall_max = population[np.argmax(weightings)]
            weightings = weightings/np.sum(weightings)
            weighted_population = np.concatenate((weightings, population), axis=1)
            population = run_locals["mutate_function"](self.cross(weighted_population, run_locals), run_locals["axes"])
        return overall_max








