import random as rd
import numpy as np
import math
import copy
import time
import UnitTests

GLOBAL_TIME_UPDATE_FREQUENCY = 5  # update frequency for specific verbosity outputs, (needs a better explanation.)


class ParticleSwarmOptimization:
    """
    Class optimization by particle swarm
    """

    # locals are stored in a dict because then they can be set
    # for a general case in addition to specific run by run cases
    locals = {  # Set of local variables for running the optimizer
        "fitness_function": None,  # Function to be optimized: POINT -> VALUE
        "population_size": -1,  # How large should the population be

        # The ending condition chooses how long the algorithm will run for TBC
        "end_condition": "time_constraint",  # What ending condition to use
        "time_constraint": 120,  # How long should the algorithm run
        "generations": 50,  # How many generations to run the algorithm for

        "axes": None,  # (list(tuple(float))) What are the axes of the algorithm
        "weighting_bias": lambda x: x,  # Bias function for weighting... NOT IMPLEMENTED YET
        "PSO_VELOCITY_WEIGHT": .5,  # Particle swarm optimization velocity weight
        "PSO_INDIVIDUAL_WEIGHT": .2,  # Particle swarm optimization individual's highest found point weight
        "PSO_GROUP_WEIGHT": .3,  # Particle swarm optimization group's highest found point weight

        # not implemented yet:
        # should probably make the same thing for starting positions, (use the 1d graph as a pdf?)
        "starting_velocity_ranges": None,  # The starting velocity ranges of the
        "starting_velocity_function": None,  # (int -> list(tuple(float)) -> list(tuple(float)) -> np.array(float))
        # function that takes in the size of the population and the velocity ranges and outputs the starting velocities
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
        self.testing_unit = UnitTests.ParticleSwarmUnitTests(testing_level=testing_level, verbosity=testing_verbosity)

        for key, val in params.items():
            self.set(key, val)  # invoke set so that all continuous checking for changed parameters happens only once
            # place

    def set(self, param, value):
        """
        Sets the value of a parameter

        Args:
            param (int): String of parameter to set
            value (obj): String of value for parameter to be set to. NB: to make phi = "lambda". value must be ""lambda""

        Returns:
            int: 0 upon failure, 1 if succeeded
        """
        # continuous testing of inputs
        if self.testing_unit.testing_level > 1 and not self.testing_unit.c_test_set_inp(param, value):
            raise ValueError("set won't run, input's aren't valid.")

        # continuous testing of functional inputs
        if self.testing_unit.testing_level > 0:
            if param in ["weighting_bias"]:
                    if not [self.testing_unit.c_test_weighting_bias][["weighting_bias"].index(param)](value):
                        raise ValueError("Bad " + param + " input. See log or raise testing verbosity.")

        locals[param] = value  # Security Risk
        return 1  # Success

    def get_locals_copy(self):  # this function has none of its own testing because of its simplicity
        """
        Returns a copy of the locals for this instance

        Returns:
            dict: copy of self.locals
        """
        return copy.deepcopy(self.locals)  # a copy is made so no changes propagate after function call

    def step(self, particles, best_state, best_fitness, run_locals):
        """
        Helper function of run. Does inner loop.

        Args:
            particles (np.array): information on current position and velocity of particles
            best_state (np.array): information on the best position of particles
            best_fitness (np.array): information on the fitness of particles best position
            run_locals (dict): local run by run information

        returns:
            particles (np.array): information on current position and velocity of particles after this step
            best_state (np.array): information on the best position of particles after this step
            best_fitness (np.array): information on the fitness of particles best position after this step
        """
        # continuous testing of inputs
        if self.testing_unit.testing_level > 1 and not self.testing_unit.c_test_step_inp(particles,
                                                                                           best_state,
                                                                                           best_fitness,
                                                                                           run_locals):
            raise ValueError("step won't run, input's aren't valid.")
        # apply the fitness function to get this generations fitness values
        fitness = np.apply_along_axis(run_locals["fitness_function"], 1, particles[:, 0, :])

        # find any personal improvements
        better = best_fitness < fitness
        # set them
        best_fitness[better] = fitness[better]
        # set their states
        best_state[better] = particles[better, 0]

        # find highest of group
        best_of_group = np.argmax(best_fitness, axis=0)

        if self.verbosity > 6:  # some random high verbosity outputs that were once used for debugging, might give ideas
            print(particles[0])
            print(particles[:, 1].shape)
            print(best_state.shape)
            print(np.repeat(best_state[best_of_group][np.newaxis, :], particles[:, 1].shape[0], axis=0).shape)

        # run calculation for the velocity calculation
        # Maurice Clerc. Standard Particle Swarm Optimisation. 2012. hal-00764996
        particles[:, 1] = (run_locals["PSO_VELOCITY_WEIGHT"] * particles[:, 1] +
                           run_locals["PSO_INDIVIDUAL_WEIGHT"] * rd.random() * (best_state - particles[:, 1]) +
                           run_locals["PSO_GROUP_WEIGHT"] * rd.random() * (np.repeat(best_state[best_of_group][np.newaxis, :],
                                                                                     particles[:, 1].shape[0], axis=0)
                                                                           - particles[:, 1]))

        # run calculation for point calculation
        particles[:, 0] = particles[:, 0] + particles[:, 1]
        return particles, best_state, best_fitness

    def run(self, temp_params={}):
        """
        Runs the particle swarm optimization.
        Standard implementation mostly based on: Maurice Clerc. Standard Particle Swarm Optimisation. 2012. hal-00764996
        Some additions for more options (such as different starting pdfs) have been added. (/ being worked on)

        Args:
            temp_params (obj:'dict', optional): parameters set for this run only

        Returns:
            np.array: choice with maximum value of all particles
        """
        # continuous testing of inputs
        if self.testing_unit.testing_level > 1 and not self.testing_unit.c_test_step_inp(temp_params, self.locals):
            raise ValueError("run won't run, input's aren't valid.")

        # continuous testing of functional inputs
        if self.testing_unit.testing_level > 0:
            for key, val in temp_params.items():
                if key in ["population_function", "mutate_function", "cross_function", "weighting_bias"]:
                    if not [self.testing_unit.c_test_weighting_bias][["weighting_bias"].index(key)](val):
                        raise ValueError("Bad " + key + " input. See log or raise testing verbosity.")

        # set the single run locals
        run_locals = copy.deepcopy(self.locals)
        for param, value in temp_params.items():
            run_locals[param] = value

        # initialize all arrays
        # current state of particles
        particles = np.full((run_locals["population_size"], 2, len(run_locals["axes"])), -1, dtype=float)
        # best fitness value achieved
        best_fitness = np.zeros((run_locals["population_size"]))
        # state that each particle is when it found its best fitness value
        best_state = np.full((run_locals["population_size"], len(run_locals["axes"])), -1, dtype=float)
        # initialize particles numbers
        for j in range(particles.shape[2]):
            particles[:, 0, j] = np.random.uniform(low=run_locals["axes"][j][0], high=run_locals["axes"][j][1], size=particles.shape[0])
            if not run_locals["starting_velocity_ranges"]:
                particles[:, 1, j] = 0  # zero starting velocity
            else:
                raise NotImplementedError("Functional starting velocities aren't implemented yet")

        # this generally shouldn't be a program, can be removed after some use where this value error doesn't appear
        if self.testing_unit.testing_level > 1:  # the rare in implementation testing
            for i in particles.flatten():
                if i == -1:
                    raise ValueError("This program didn't properly initialize its particles array")

        # split into different termination criteria
        if run_locals["end_condition"] == "time_constraint":
            start_time = time.time()  # initial starting time
            last_update = time.time()  # last time an update was printed (for verbosity > 1 only)

            # start the main loop
            while time.time() < start_time + run_locals["time_constraint"]:
                # some outputs based on verbosity
                if self.verbosity > 3:
                    print(particles)
                    print(best_state)
                    print(best_fitness)
                elif self.verbosity > 1:
                    if time.time() < last_update + GLOBAL_TIME_UPDATE_FREQUENCY:
                        print("update")
                        last_update = time.time()

                # run the step function
                particles, best_state, best_fitness = self.step(particles, best_state, best_fitness, run_locals)

            # once done return the best state
            return best_state[np.argmax(best_fitness, axis=0)]
        elif run_locals["end_condition"] == "generations":
            for _ in range(run_locals["generations"]):
                # some outputs based on verbosity
                if self.verbosity > 3:
                    print(particles)
                    print(best_state)
                    print(best_fitness)
                elif self.verbosity > 1:
                    print("update")

                # run the step function
                particles, best_state, best_fitness = self.step(particles, best_state, best_fitness, run_locals)

            # once done return the best state
            return best_state[np.argmax(best_fitness, axis=0)]
        else:
            raise ValueError("This line should never be reached")





# not well implemented nor tested
class MetaParticleSwarmOptimization:
    """
    Class for meta optimization upon ParticleSwarmOptimization class
    """
    base_params = ()
    axes = None

    def __init__(self, d=1):
        """
        Initializes a GeneticAlgorithm class instance

        Args:
            d (int): depth of meta optimization
        """
        self.depth = d

    def set_depth(self, d):
        """
        Sets the depth of the meta optimization

        Args:
            d (int): depth of meta optimization
        """
        self.depth = d

    def set_axes(self, a):
        """
        Sets the axes of the final optimization

        Args:
            a (list(tuple)): axes
        """
        self.axes = a

    def run(self, fitness_function):
        """
        Runs the meta optimization with a fitness_function

        Args:
            fitness_function (np.array -> value): function to maximise

        Returns:
            np.array: choice with maximum value of all particles
        """
        return run_helper(depth, fitness_function, 0, 0, 0)  # fixme

    def run_helper(self, d, fitness_function, v, i, g):
        """
        Helper for the run function, makes a recursive loop possible

        Args:
            d (int): current depth
            fitness_function (np.array -> value): function to maximise
            v (float): velocity weight
            i (float): individual weight
            g (float): group max weight

        Returns:
            np.array: choice with maximum value of all particles
        """
        if d == 0:
            pso = ParticleSwarmOptimization()
            pso.set("fitness_function", lambda a, b, c: run_helper(d-1, fitness_function, a, b, c))
            pso.set("axes", [(0, 1), (0, 1), (0, 1)])
            pso.set("PSO_VELOCITY_WEIGHT", v)
            pso.set("PSO_INDIVIDUAL_WEIGHT", i)
            pso.set("PSO_GROUP_WEIGHT", g)
            return pso.run()
        else:
            pso = ParticleSwarmOptimization()
            pso.set("fitness_function", fitness_function)
            pso.set("axes", axes)
            pso.set("PSO_VELOCITY_WEIGHT", v)
            pso.set("PSO_INDIVIDUAL_WEIGHT", i)
            pso.set("PSO_GROUP_WEIGHT", g)
            return pso.run()
