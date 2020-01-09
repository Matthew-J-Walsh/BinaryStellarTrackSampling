import numpy as np
import copy
import time
from . import UnitTests

"""
NB: These functions are not secure, anyone with access to them will have access to the functions permission level.
If you are concerned about security do an analysis of all the code before using.

MUTATION_SIGMA = .3
"""


def random_population(count, points, axes):
    """
    Computes a random population of size (count) with axes specified.

    Args:
        count (int): amount of items in population
        points (int): number of points along each axis
        axes (list(tuple(float))): list or tuple of length 2 lists or tuples. Each of these
        tuples/lists define the lower and upper bound on their respective
        axes ex: [(0, 10), (0, 5)] would mean that there are two axes, on
        from 0 to 10 and the other from 0 to 5

    Returns:
        np.array: A random population from the input specifications
    """
    population = np.empty((count, points, len(axes)))
    for i in range(count):
        for k in range(len(axes)):
            population[i, :, k] = np.random.uniform(low=axes[k][0], high=axes[k][1], size=points)
    return population


def analogue_random_population(count, axes):
    """
    Computes a random population of size (count) with axes specified. Will only pick values inside the axes, the not range
    of the axes. This function should be used for non-continuous functions

    Args:
        count (int): amount of items in population
        axes (list(tuple(float))): list or tuple of indiscriminate length lists or tuples. Each of
        these describe all inputs allowed along said axis
        axes ex: TBC

    Returns:
        np.array: A random population from the input specifications
    """
    population = np.empty((count, len(axes)))
    for j in range(len(axes)):
        population[:, j] = np.random.choice(axes[j], size=count)
    return population


def mutate(population, point_count, axes, generations=0,
           MUTATION_CHANCE=.3, MUTATION_DECREASE_PER_GENERATION=0):
    """
    Computes a mutation each member of a population.

    Args:
        population (np.array): population to be mutated
        axes (list(tuple(float))): axes of the population
        generations (obj:'int', optional): number of generations into genetic algorithm
        MUTATION_CHANCE (obj:'float', optional): chance at a mutation happening
        MUTATION_DECREASE_PER_GENERATION (obj:'float', optional): decreased chance of mutation per generation

    Returns:
        np.array: a population that has been randomly mutated that still lies within the axes specified.
    """
    newdna = np.empty(population.shape)
    for i in range(population.shape[0]):
        for j in range(population.shape[1]):
            for k in range(population.shape[2]):
                # uniformly pick a new point across axis
                c = population[i][j][k]
                if np.random.rand() < MUTATION_CHANCE - MUTATION_DECREASE_PER_GENERATION * generations:
                    c = np.random.uniform(axes[k][0], axes[k][1], size=1)[0]
                newdna[i][j][k] = c
    return newdna


def analogue_mutate(population, axes, generations=0,
           MUTATION_CHANCE=.3, MUTATION_DECREASE_PER_GENERATION=0):
    """
    Computes a mutation each member of a non-continuous population.

    Args:
        population (np.array): population to be mutated
        axes (list(tuple(float))): axes of the population
        generations (obj:'int', optional): number of generations into genetic algorithm
        MUTATION_CHANCE (obj:'float', optional): chance at a mutation happening
        MUTATION_DECREASE_PER_GENERATION (obj:'float', optional): decreased chance of mutation per generation

    Returns:
        np.array: a population that has been randomly mutated that still lies within the axes specified.
    """
    newdna = np.empty(population.shape)
    for i in range(population.shape[0]):
        for j in range(population.shape[1]):
            c = population[i][j]
            # simply random choice in allowed axis
            if np.random.rand() < MUTATION_CHANCE - MUTATION_DECREASE_PER_GENERATION * generations:
                c = np.random.choice(axes[j], size=1)[0]
            newdna[i][j] = c
    return newdna


def my_gauss(val, axis):
    """
    TBC
    """
    newval = np.random.normal(loc=val, scale=MUTATION_SIGMA)
    return max(min(newval, axis[1]), axis[0])


def functional_mutation(population, axes, function=my_gauss):
    """
    Mutates a population based on a function

    Args:
        population (np.array): population to be mutated
        axes (list(tuple(float))): axes of the population
        function (obj: np.array -> list(tuples) -> np.array): Individual -> axes -> Individual

    Returns:
        np.array: A population that has been mutated based on the function given.
    """
    newdna = np.empty(population.shape)
    for i in range(population.shape[0]):
        for j in range(population.shape[1]):
            # apply the function to each set in population
            c = population[i][j]
            axis = axes[j]
            c = function(c, axis)
            newdna[i][j] = c
    return newdna


def cross(a, b, CROSSOVER_CHANCE=.3):
    """
    Calculates a genetic crossover between two strands

    Args:
        a (np.array): first dna sequence
        b (np.array): second dna sequence

    Returns:
        np.array: their (random) cross
    """
    c = np.empty((a.shape[0]))
    for i in range(a.shape[0]):
        # either can one, the other, or a combination
        if np.random.rand() < CROSSOVER_CHANCE:
            if np.random.rand() > .5:
                c[i] = a[i]
            else:
                c[i] = b[i]
        else:
            c[i] = (a[i] + b[i]) / 2
    return c


def analogue_cross(a, b):
    """
    Calculates a genetic crossover between two strands (analogue)

    Args:
        a (np.array): first dna sequence
        b (np.array): second dna sequence

    Returns:
        np.array: their (random) cross
    """
    c = np.empty((a.shape[0]))
    for i in range(a.shape[0]):
        # no opportunity to combine
        if np.random.rand() > .5:
            c[i] = a[i]
        else:
            c[i] = b[i]
    return c


def gaussiancross(dna1, dna2, axes = None):
    """
    Calculates a genetic crossover between two strands

    Args:
        dna1 (np.array): first dna sequence
        dna2 (np.array): second dna sequence
        axes (None): omit

    Returns:
        returns: their (gaussian) cross
    """
    c = np.empty(dna1.shape)
    for i in range(dna1.shape[0]):
        for j in range(dna2.shape[1]):
            # gaussian between the two points, cut off at either point
            c[i][j] = max(min(np.random.normal(loc=(dna1[i][j] + dna2[i][j]) / 2, scale=abs(dna1[i][j] - dna2[i][j]) / 5),
                              max(dna1[i][j], dna2[i][j])),
                          min(dna1[i][j], dna2[i][j]))
    return c


def nearestpoint(point, axis):
    """
    Calculates the nearest of a continuous number along a non-continuous axis. Helper function to analogue_gaussiancross

    Args:
        point (np.array): continuous point
        axis (tuple(float)): axis along which the nearest should be found

    Returns:
        np.array: point along axis nearest to "point"
    """
    return axis[np.argmin(abs(np.array(axis)-point))]


def analogue_gaussiancross(dna1, dna2, axes=None):
    """
    Calculates a genetic crossover between two strands

    Args:
        dna1 (np.array): first dna sequence
        dna2 (np.array): second dna sequence
        axes (TBC): axes of the dna

    Returns:
        np.array: their (gaussian) cross on a non-continuous (shape?)
    """
    # cross
    c = gaussiancross(dna1, dna2)

    # then find the nearest valid point
    for i in range(len(c)):
        c[i] = nearestpoint(c[i], axes[i])
    return c


class PopulationAlgorithm:
    """
    Class for use of Population Genetic Algorithm

    SECURITY WARNING: THIS IS NOT A SECURE IMPLEMENTATION, DON'T GIVE THIS FUNCTION ANY PERMISSIONS YOU DON'T WANT END USER TO HAVE
    """

    # locals are stored in a dict because then they can be set
    # for a general case in addition to specific run by run cases
    __locals = {  # Set of local variables for running the optimizer
        "fitness_function": None,  # (np.array -> np.array -> float) Function to be optimized
        "population_size": -1,  # (int) How large should the population be

        # The ending condition chooses how long the algorithm will run for TBC
        "end_condition": "time_constraint",  # What ending condition to use
        "time_constraint": 120,  # How long should the algorithm run
        "generations": 50,  # How many generations to run the algorithm for

        "axes": None,  # (list(tuple(float))) What are the axes of the algorithm
        "population_function":
            random_population,  # (int -> list(tuple(float))) -> np.array)
        # What function should be used to create a random population
        "mutate_function": functional_mutation,  # (np.array -> list(tuple(float))) -> np.array) The mutation function
        "cross_function": cross,  # (np.array -> np.array -> np.array) The crossover function
        "weighting_bias": lambda x: x,  # (float -> float) Bias function for weighting

        "seed": None,  # random seed for each run. if none a fresh random seed will be generated each time
    }

    def __init__(self, params={}, verbosity=0, testing_level=1, testing_verbosity=1,
                 test_functions=False):
        """
        Initializes a GeneticAlgorithm class instance

        Args:
            params (obj:'dict', optional): dictionary of parameters to set
            verbosity (int [0,+): verbosity level
            level 0: no internal prints <----- suggested
            level 1: temporary verbosity only. No permanent code should have this verbosity.
            level 2: basic verbosity, output updates and basic information
            level 3: higher level verbosity, information generation by generation (if generation based)
            level 4+: reserved for higher, more spam like, outputs.
            Anything at or above 3 will noticeably adversely effect the runtime of the program.
            testing_level (int): testing level. See UnitTests.py for full documentation
            testing_verbosity (int): testing verbosity level. See UnitTests.py for full documentation
        """
        # set up unit testing for this instance of the class
        self.verbosity = verbosity
        self.testing_unit = UnitTests.PopulationUnitTests(self, testing_level=testing_level,
                                                          verbosity=testing_verbosity,
                                                          test_functions=test_functions)

        for key, val in params.items():
            self.set(key, val)  # invoke set so that all continuous checking for changed parameters happens only once
            # place

    def set(self, param, value):
        """
            Sets the value of a parameter

        Args:
            param (obj): parameter to set
            value (obj): what parameter will be set to in locals

        Returns:
            int: 0 upon failure, 1 if successful
        """
        # continuous testing of inputs
        if self.testing_unit.testing_level > 1 and not self.testing_unit.c_test_set_inp(param, value):
            raise ValueError("set won't run, input's aren't valid.")

        # continuous testing of functional inputs
        if self.testing_unit.testing_level > 0:
            if param in ["population_function", "mutate_function", "cross_function", "weighting_bias"]:
                if not [self.testing_unit.c_test_population_function,
                        self.testing_unit.c_test_mutate_function,
                        self.testing_unit.c_test_cross_function,
                        self.testing_unit.c_test_weighting_bias]\
                        [["population_function", "mutate_function",
                          "cross_function", "weighting_bias"].index(param)](value):
                    raise ValueError("Bad " + param + " input. See log or raise testing verbosity.")

        self.__locals[param] = value  # Security Risk
        return 1  # success

    def get_locals_copy(self):  # this function has none of its own testing because of its simplicity
        """
        Returns a copy of the locals for this instance

        Returns:
            dict: copy of self.locals
        """
        return copy.deepcopy(self.__locals)  # a copy is made so no changes propagate after 'set' calls

    def _cross(self, weighted_population, run_locals):
        """
        Completes cross on entire space based on weightings

        Args:
            weighted_population (np.array): population with weighting summing to 1
            run_locals (dict): local run information

        Returns:
            returns: the new population made from crossing
        """
        # continuous testing of inputs
        if self.testing_unit.testing_level > 1 and not self.testing_unit.c_test__cross_inp(weighted_population,
                                                                                           run_locals):
            raise ValueError("_cross won't run, input's aren't valid.")

        # initialize new population
        new_population = np.empty((weighted_population.shape[0], weighted_population.shape[1] - 1))

        # use the cross function to fill in the new population
        for i in range(weighted_population.shape[0]):
            new_population[i] = run_locals["cross_function"](weighted_population[:, 1:][np.random.choice(
                                                                 np.arange(weighted_population.shape[0]),
                                                                 p=weighted_population[:, 0])],
                                                             weighted_population[:, 1:][np.random.choice(
                                                                 np.arange(weighted_population.shape[0]),
                                                                 p=weighted_population[:, 0])])
        return new_population

    def run(self, temp_params={}):
        """
        Runs the genetic algorithm.

        Args:
            temp_params (dict): parameters set for this run only

        Returns:
            np.array: final population
        """
        # continuous testing of inputs
        if self.testing_unit.testing_level > 1 and not self.testing_unit.c_test_run_inp(temp_params, self.__locals):
            raise ValueError("run won't run, input's aren't valid.")

        # continuous testing of functional inputs
        if self.testing_unit.testing_level > 0:
            for key, val in temp_params.items():
                if key in ["population_function", "mutate_function", "cross_function", "weighting_bias"]:
                    if not [self.testing_unit.c_test_population_function,
                            self.testing_unit.c_test_mutate_function,
                            self.testing_unit.c_test_cross_function,
                            self.testing_unit.c_test_weighting_bias]\
                            [["population_function", "mutate_function",
                              "cross_function", "weighting_bias"].index(key)](val):
                        raise ValueError("Bad " + key + " input. See log or raise testing verbosity.")

        # set the single run locals
        run_locals = copy.deepcopy(self.__locals)
        for param, value in temp_params.items():
            run_locals[param] = value

        # sets random seed
        if run_locals["seed"] is None:
            np.random.seed()
        else:
            np.random.seed(run_locals["seed"])

        # initialize population
        population = run_locals["population_function"](run_locals["population_size"], run_locals["axes"])

        # split into different termination criteria
        if run_locals["end_condition"] == "time_constraint":
            start_time = time.time()
            while time.time() < start_time + run_locals["time_constraint"]:
                population = self.run_helper(population, run_locals)
            # return final population
            return population
        elif run_locals["end_condition"] == "generations":
            for _ in range(run_locals["generations"]):
                population = self.run_helper(population, run_locals)
            # return final population
            return population
        else:
            raise ValueError("This line should never be reached")

    def run_helper(self, population, run_locals):
        """
        Helper function for run. Auto-generated.

        Args:
            population: See run
            run_locals: See run

        Returns:
            See run
        """
        # apply fitness function
        weightings = np.apply_along_axis(run_locals["fitness_function"], 1, population, population) \
            .reshape((population.shape[0], 1))
        # apply bias
        weightings = np.apply_along_axis(run_locals["bias_function"], 1, weightings)  # not sure if should be 1 or 0 !!!!!!!!!!!!!!!!!!!!!!!!
        # make sure there are no values below zero (can't have negative probabilities)
        weightings[weightings < 0] = 0
        # normalize weightings
        weightings = weightings / np.sum(weightings)
        # place population and weightings together for ease
        weighted_population = np.concatenate((weightings, population), axis=1)
        # do cross then mutation
        population = run_locals["mutate_function"](self._cross(weighted_population, run_locals),
                                                   run_locals["axes"])
        return population


class GeneticAlgorithm:
    """
    Class for use of Genetic Algorithm

    SECURITY WARNING: THIS IS NOT A SECURE IMPLEMENTATION, DON'T GIVE THIS FUNCTION ANY PERMISSIONS YOU DON'T WANT CALLER TO HAVE
    """

    # locals are stored in a dict because then they can be set
    # for a general case in addition to specific run by run cases
    __locals = {  # Set of local variables for running the optimizer
        "fitness_function": None,  # (np.array -> float) Function to be optimized
        "population_size": -1,  # (int) How large should the population be

        # The ending condition chooses how long the algorithm will run for TBC
        "end_condition": "time_constraint",  # What ending condition to use
        "time_constraint": 120,  # How long should the algorithm run
        "generations": 50,  # How many generations to run the algorithm for

        "axes": None,  # (list(tuple(float))) What are the axes of the algorithm
        "population_function":
            random_population,  # (int -> list(tuple(float))) -> np.array)
        # What function should be used to create a random population
        "mutate_function": mutate,  # (np.array -> list(tuple(float))) -> np.array) The mutation function
        "cross_function": gaussiancross,  # (np.array -> np.array -> np.array) The crossover function
        "point_count": 1,  # How many points to use
        "weighting_bias": lambda x: x,  # (float -> float) Bias function for weighting... NOT IMPLEMENTED YET

        "seed": None,  # random seed for each run. if none a fresh random seed will be generated each time
    }

    def __init__(self, params={}, verbosity=0, testing_level=1, testing_verbosity=1):
        """
        Initializes a GeneticAlgorithm class instance

        Args:
            params (obj:'dict', optional): dictionary of parameters to set
            verbosity (int [0,+): verbosity level
            level 0: no internal prints <----- suggested
            level 1: temporary verbosity only. No permanent code should have this verbosity.
            level 2: basic verbosity, output updates and basic information
            level 3: higher level verbosity, information generation by generation (if generation based)
            level 4+: reserved for higher, more spam like, outputs.
            Anything at or above 3 will noticeably adversely effect the runtime of the program.
            testing_level (int): testing level. See UnitTests.py for full documentation
            testing_verbosity (int): testing verbosity level. See UnitTests.py for full documentation
        """
        self.verbosity = verbosity
        self.testing_unit = UnitTests.GeneticUnitTests(testing_level=testing_level, verbosity=testing_verbosity)

        for key, val in params.items():
            self.set(key, val)  # invoke set so that all continuous checking for changed parameters happens only once
            # place

    def set(self, param, value):
        """
        Sets the value of a parameter

        Args:
            param (obj): parameter to set
            value (obj): what parameter will be set to.

        Returns:
            int: 0 upon failure, 1 if successful
        """
        # continuous testing of inputs
        if self.testing_unit.testing_level > 1 and not self.testing_unit.c_test_set_inp(param, value):
            raise ValueError("set won't run, input's aren't valid.")

        # continuous testing of functional inputs
        if self.testing_unit.testing_level > 0:
            if param in ["fitness_function", "population_function", "mutate_function", "cross_function", "weighting_bias"]:
                if not [self.testing_unit.c_test_fitness_function,
                        self.testing_unit.c_test_population_function,
                        self.testing_unit.c_test_mutate_function,
                        self.testing_unit.c_test_cross_function,
                        self.testing_unit.c_test_weighting_bias]\
                        [["fitness_function", "population_function", "mutate_function",
                          "cross_function", "weighting_bias"].index(param)](value):
                    raise ValueError("Bad " + param + " input. See log or raise testing verbosity.")

        self.__locals[param] = value  # Security Risk
        return 1

    def get_locals_copy(self):  # this function has none of its own testing because of its simplicity
        """
        Returns a copy of the locals for this instance

        Returns:
            dict: copy of self.locals
        """
        return copy.deepcopy(self.__locals)  # a copy is made so no changes propagate after function call

    def _cross(self, old_population, population_weighting, run_locals):
        """
        Completes cross on entire space based on weightings

        Args:
            old_population (np.array): population
            population_weighting (np.array): weighting for each item in population summing to 1
            run_locals (dict): local run information

        Returns:
            np.array: the new population made from crossing
        """
        # continuous testing of inputs
        if self.testing_unit.testing_level > 1 and not self.testing_unit.c_test__cross_inp(old_population,
                                                                                           population_weighting,
                                                                                           run_locals):
            raise ValueError("_cross won't run, input's aren't valid.")

        # initialize new population
        new_population = np.empty(old_population.shape)

        # use the cross function to fill in the new population
        for i in range(old_population.shape[0]):  # maybe can do this with a apply_along_axis
            new_population[i] = run_locals["cross_function"](old_population[np.random.choice(
                                                                 np.arange(old_population.shape[0]),
                                                                 p=population_weighting)],
                                                             old_population[np.random.choice(
                                                                 np.arange(old_population.shape[0]),
                                                                 p=population_weighting)],
                                                             run_locals["axes"])
        return new_population

    def eval(self, population, run_locals):
        """
        Evaluates a population and returns their weights

        Args:
            population (np.array): population
            run_locals (dict): local run information

        Returns:
            np.array: th weightings of the population
        """
        # continuous testing of inputs
        if self.testing_unit.testing_level > 1 and not self.testing_unit.c_test_eval_inp(population, run_locals):
            raise ValueError("eval won't run, input's aren't valid.")

        # initialize weightings
        weightings = np.empty((population.shape[0]))

        # for loop is used rather than any numpy operations as this function applies a fitness function across
        # multidimensional cross sections of the population, np.apply_along_axis only applies a function across a
        # 1d slice
        for i in range(population.shape[0]):
            weightings[i] = run_locals["fitness_function"](population[i])

        # no weightings < 0 are allowed as no negative probabilities are allowed
        weightings[weightings < 0] = 0
        return weightings

    def run(self, temp_params={}):
        """
        Runs the genetic algorithm

        Args:
            temp_params (dict): parameters set for this run only

        Returns:
            np.array: overall max set
        """
        # continuous testing of inputs
        if self.testing_unit.testing_level > 1 and not self.testing_unit.c_test_run_inp(temp_params, self.__locals):
            raise ValueError("run won't run, input's aren't valid.")

        # continuous testing of functional inputs
        if self.testing_unit.testing_level > 0:
            for key, val in temp_params.items():
                if key in ["fitness_function", "population_function", "mutate_function", "cross_function", "weighting_bias"]:
                    if not [self.testing_unit.c_test_fitness_function,
                            self.testing_unit.c_test_population_function,
                            self.testing_unit.c_test_mutate_function,
                            self.testing_unit.c_test_cross_function,
                            self.testing_unit.c_test_weighting_bias]\
                            [["fitness_function", "population_function", "mutate_function",
                              "cross_function", "weighting_bias"].index(key)](val):
                        raise ValueError("Bad "+key+" input. See log or raise testing verbosity.")

        # set the single run locals
        run_locals = copy.deepcopy(self.__locals)
        for param, value in temp_params.items():
            run_locals[param] = value

        if run_locals["seed"] is None:
            np.random.seed()
        else:
            np.random.seed(run_locals["seed"])

        # initialize population via functional input
        population = run_locals["population_function"](run_locals["population_size"],
                                                       run_locals["point_count"],
                                                       run_locals["axes"])
        overall_max = None
        overall_max_val = 0

        if run_locals["end_condition"] == "time_constraint":
            start_time = time.time()
            while time.time() < start_time + run_locals["time_constraint"]:
                population, overall_max, overall_max_val = self.run_helper(population,
                                                                           overall_max,
                                                                           overall_max_val,
                                                                           run_locals)
            # return the max found
            return overall_max
        elif run_locals["end_condition"] == "generations":
            while _ in range(run_locals["generations"]):
                population, overall_max, overall_max_val = self.run_helper(population,
                                                                           overall_max,
                                                                           overall_max_val,
                                                                           run_locals)
            # return the max found
            return overall_max
        else:
            raise ValueError("This line should never be reached")

    def run_helper(self, population, overall_max, overall_max_val, run_locals):
        """
        Helper function for run. Auto-generated (mostly).

        Args:
            population: See run
            overall_max: See run
            overall_max_val: See run
            run_locals: See run

        Returns:
            See run
        """
        weightings = self.eval(population, run_locals)
        # update the maximum
        if overall_max_val < weightings.max():
            overall_max_val = weightings.max()
            overall_max = population[np.argmax(weightings)]
        # normalize
        weightings = weightings / np.sum(weightings)
        # cross and mutate
        population = run_locals["mutate_function"](self._cross(population, weightings, run_locals),
                                                   run_locals["point_count"], run_locals["axes"])
        return population, overall_max, overall_max_val








