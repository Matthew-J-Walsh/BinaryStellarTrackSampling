import random as rd
import numpy as np
import math
import copy
import time





"""
Class: ParticleSwarmOptimization
------------------------------------
Class optimization by particle swarm
"""
class ParticleSwarmOptimization:
    locals = {
        "fitness_function": None,  # Function to be optimized: POINT -> VALUE
        "population_size": -1,  # How large should the population be
        "time_constraint": 120,  # How long should the algorithm run
        "axes": None,  # What are the axes of the algorithm
        "weighting_bias": lambda x: x,  # Bias function for weighting... NOT IMPLEMENTED YET
        "PSO_VELOCITY_WEIGHT": .5,
        "PSO_INDIVIDUAL_WEIGHT": .2,
        "PSO_GROUP_WEIGHT": .3,
    }

    """
    Function: ParticleSwarmOptimization.__init__
    --------------------------------------------
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
    Function: ParticleSwarmOptimization.set
    ------------------------------
    Sets the value of a parameter

    param: String of parameter to set
    value: String of value for parameter to be set to. NB: to make phi = "lambda". value must be ""lambda""

    returns 0 upon failure, 1 if succeeded
    """

    def set(self, param, value):
        try:
            locals[param] = value  # Security Risk
        except:
            return 0
        return 1

    """
    Function: ParticleSwarmOptimization.step
    ----------------------------------------
    Helper function of run. Does inner loop.

    particles: information on current position and velocity of particles
    best_state: information on the best position of particles
    best_fitness: information on the fitness of particles best position
    run_locals: local run by run information
    
    returns: 0 under failure condition, otherwise 1. Values of inputs are changed
    """
    def step(self, particles, best_state, best_fitness, run_locals):
        fitness = np.apply_along_axis(run_locals["fitness_function"], 1, particles[:, 0, :])
        better = best_fitness < fitness
        best_fitness[better] = fitness[better]
        best_state[better] = particles[better, 0]
        best_of_group = np.argmax(best_fitness, axis=0)
        #print(particles[0])
        #print(particles[:, 1].shape)
        #print(best_state.shape)
        #print(np.repeat(best_state[best_of_group][np.newaxis, :], particles[:, 1].shape[0], axis=0).shape)
        particles[:, 1] = (run_locals["PSO_VELOCITY_WEIGHT"] * particles[:, 1] +
                           run_locals["PSO_INDIVIDUAL_WEIGHT"] * rd.random() * (best_state - particles[:, 1]) +
                           run_locals["PSO_GROUP_WEIGHT"] * rd.random() * (np.repeat(best_state[best_of_group][np.newaxis, :],
                                                                                     particles[:, 1].shape[0], axis=0)
                                                                           - particles[:, 1]))
        particles[:, 0] = particles[:, 0] + particles[:, 1]
        return particles, best_state, best_fitness

    """
    Function: ParticleSwarmOptimization.run
    ---------------------------------------
    Runs the particle swarm optimization.

    temp_params: parameters set for this run only

    returns: choice with maximum value of all particles
    """
    def run(self, temp_params={}):
        run_locals = copy.deepcopy(self.locals)
        for param, value in temp_params.items():
            run_locals[param] = value
        particles = np.full((run_locals["population_size"], 2, len(run_locals["axes"])), -1, dtype=float)
        best_fitness = np.zeros((run_locals["population_size"]))
        best_state = np.full((run_locals["population_size"], len(run_locals["axes"])), -1, dtype=float)
        for j in range(particles.shape[2]):
            particles[:, 0, j] = np.random.uniform(low=run_locals["axes"][j][0], high=run_locals["axes"][j][1], size=particles.shape[0])
            particles[:, 1, j] = 0 #zero starting velocity
        start_time = time.time()
        while time.time() < start_time + run_locals["time_constraint"]:
            #print(particles)
            particles, best_state, best_fitness = self.step(particles, best_state, best_fitness, run_locals)
        #print(best_state)
        #print(best_fitness)
        return best_state[np.argmax(best_fitness, axis=0)]





"""
Class: MetaParticleSwarmOptimization
------------------------------------
Class for meta optimization upon ParticleSwarmOptimization class
"""
class MetaParticleSwarmOptimization:
    depth = -1
    base_params = ()
    axes = None

    """
    Function: MetaParticleSwarmOptimization.__init__
    --------------------------------------------
    Initializes a GeneticAlgorithm class instance

    d: depth of meta optimization
    """
    def __init__(self, d=1):
        self.depth = d

    """
    Function: MetaParticleSwarmOptimization.set_depth
    --------------------------------------------
    Sets the depth of the meta optimization

    d: depth of meta optimization
    """
    def set_depth(self, d):
        self.depth = d

    """
    Function: MetaParticleSwarmOptimization.set_axes
    --------------------------------------------
    Sets the axes of the final optimization

    a: axes
    """
    def set_axes(self, a):
        self.axes = a

    """
    Function: MetaParticleSwarmOptimization.run
    --------------------------------------------
    Runs the meta optimization with a fitness_function

    fitness_function: function to maximise
    """
    def run(self, fitness_function):
        run_helper(depth, fitness_function)

    """
    Function: MetaParticleSwarmOptimization.run_helper
    --------------------------------------------
    Helper for the run function, makes a recursive loop possible

    d: current depth
    fitness_function: function to maximise
    v: velocity weight
    i: individual weight
    g: group max weight
    """
    def run_helper(self, d, fitness_function, v, i, g):
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
