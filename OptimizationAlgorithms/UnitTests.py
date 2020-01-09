from . import GeneticAlgorithms
from . import ParticleSwarm
import numpy as np
import string


def general_weighting_bias(verbosity, function):
    """
    General case for weighting bias. See any uses in classes.

    Args:
        function: See uses.
        verbosity: See uses.

    Returns:
        See uses.
    """
    l = np.random.random(size=1000) * 100
    r = np.apply_along_axis(function, 0, l)
    if (r.dtype.kind not in {'b', 'u', 'i', 'f', 'c'}) or (r.shape[1] > 1):
        if verbosity > 0:
            print("ERROR: weighting bias returned incorrect array or type")
        return 0
    else:
        if verbosity > 1:
            print("Weighting bias function returns correctly")
        return 1


def general_cross_function(verbosity, function):
    """
    General case for cross function. See any uses in classes.

    Args:
        verbosity: See uses.
        function: See uses.

    Returns:
        See uses.
    """
    ret = 1
    first_errors = [False, False]
    for count in range(10, 25, 5):
        for points in range(5, 10):
            for ax_c in range(3, 5):
                axes = []
                for _ in range(ax_c):
                    axes.append(((np.random.random_sample() * 2), (3 + np.random.random_sample() * 4)))
                population = GeneticAlgorithms.random_population(count, points, axes)  # assumes this works
                for _ in range(len(population)):
                    rd1 = np.random.choice(population)
                    rd2 = np.random.choice(population)
                    crs = function(rd1, rd2)
                    if crs.shape != rd1.shape:
                        ret = 0
                        if verbosity > 0 and first_errors[0]:
                            first_errors[0] = True
                            print("ERROR: cross function doesn't return correct shape")
                    for i in range(points):
                        for j in range(ax_c):
                            if crs[i][j] < min(rd1[i][j], rd2[i][j]) or crs[i][j] > max(rd1[i][j], rd2[i][j]):
                                ret = 0
                                if verbosity > 0 and first_errors[1]:
                                    first_errors[1] = True
                                    print("ERROR: cross function doesn't return in correct range")
        return ret


def general_fitness_function(verbosity, axes, function):
    """
    General case for fitness function. See any uses in classes.

    Args:
        verbosity: See uses.
        axes: See uses.
        function: See uses.

    Returns:
        See uses.
    """
    ret = 1
    s = []
    e = []
    for i in range(len(axes)):
        s.append(axes[i][0])
        e.append(axes[i][1])
    sr = function(np.array(s))
    er = function(np.array(e))
    if not (isinstance(sr, int) or
            isinstance(sr, float) or
            isinstance(er, int) or
            isinstance(er, float)):
        ret = 0
        if verbosity > 0:
            print("ERROR: fitness function given: " + str(function) +
                  "doesn't return single number")
    else:
        if verbosity > 1:
            print("Fitness function seems functional.")
    return ret


class PopulationUnitTests:
    """
    Class for unit tests on a Population Algorithm
    """

    def __init__(self, source, testing_level=1, verbosity=1, test_functions=False):
        """
        Initializes class, does initial tests to make sure Population Algorithm code works, and sets continuous testing
        test level

        Args:
            source (obj:'PopulationAlgorithm'): population algorithm class that made this
            testing_level (int [0, 2]): level of continuous testing to do
            level 0: no continuous testing
            level 1: only test new functions for: population, mutation, cross, and weighting <------ suggested
            level 2: test all inputs to all functions in GeneticAlgorithms
            verbosity (int [0, 3+]): verbosity level,
            level 0: no internal prints from this class
            level 1: error output only <------ suggested
            level 2: error and success outputs
            level 3+: level 2 + prints throughout running
        """
        if test_functions:
            for i in [self.test_set(),
                      self.test__cross(),
                      self.test_run(),
                      self.random_test(source)]:
                if i != 1:
                    print("WARNING: AN ERROR HAS OCCURRED IN INITIAL TESTING, THIS CLASS IS UNSTABLE.")
        self.testing_level = testing_level
        self.verbosity = verbosity

    def test_set(self):
        """
        Tests: PopulationAlgorithm.set()

        Returns:
            int: 1 if successful, 0 if not
        """
        ret = 1
        fresh = GeneticAlgorithms.PopulationAlgorithm(verbosity=0, testing_level=100, testing_verbosity=self.verbosity)
        end_sets = {}
        for _ in range(0, 10):
            param = ""
            value = ""
            for __ in range(0, 50):
                param = param + string.ascii_letters[np.random.choice(24 * 2)]
                value = value + string.ascii_letters[np.random.choice(24 * 2)]
            fresh.set(param, value)
            end_sets.update({param: value})
        first_error = False
        for key, val in end_sets.items():
            if fresh.get_locals_copy()[key] != val:
                ret = 0
                if not first_error and verbosity > 0:
                    first_error = True
                    print("ERROR: set function of Population Algorithm not functioning correctly.")
        if ret and self.verbosity > 1:
            print("Set function works correctly in Population Algorithm implementation.")
        return ret

    def test__cross(self):
        """
        Tests: PopulationAlgorithm._cross()

        Returns:
            int: 1 if successful, 0 if not
        """
        fresh = GeneticAlgorithms.PopulationAlgorithm(params={"fitness_function": (lambda x: -np.sum(x) ** 2 + 10),
                                                              "population_size": 10,
                                                              "time_constraint": 2,
                                                              "axes": [(0, 5)],
                                                              "seed": 1},
                                                      verbosity=0, testing_level=100, testing_verbosity=self.verbosity)
        ret = 1 and (fresh._cross(np.array([[.5, 1, 1],
                                            [.5, 1, 1],
                                            [0, 0, 0],
                                            [0, 0, 0],
                                            [0, 0, 0],
                                            [0, 0, 0],
                                            [0, 0, 0],
                                            [0, 0, 0],
                                            [0, 0, 0],
                                            [0, 0, 0]]), fresh.get_locals_copy())
                     == np.array([[1, 1]]*10))
        # maybe want more cases

        if ret and self.verbosity > 1:
            print("_cross runs correctly")
        if not ret and self.verbosity > 0:
            print("ERROR: _cross doesn't run correctly")
        return ret

    def test_run(self):
        """
        Tests: PopulationAlgorithm.run()

        Returns:
            int: 1 if successful, 0 if not
        """
        # There is no test for this. Run is already tested in the random test section.
        return 1

    def c_test_fitness_function(self, function, axes):
        """
        Tests if fitness function is a valid fitness function

        Returns:
            int: 1 if successful, 0 if not
        """
        return general_fitness_function(self.verbosity, axes, function)

    def c_test_population_function(self, function):
        """
        Tests if population function is a valid population function.

        Returns:
            int: 1 if successful, 0 if not
        """
        ret = 1
        first_errors = [False, False]
        for count in range(10, 25, 5):
            for points in range(5, 10):
                for ax_c in range(3, 5):
                    axes = []
                    for _ in range(ax_c):
                        axes.append((np.random.random_sample()*2), (3+np.random.random_sample()*4))
                    population = function(count, points, axes)
                    if population.shape != (count, points, ax_c):
                        if self.verbosity > 0 and not first_errors[0]:
                            print("ERROR: population function didn't output valid shape")
                            first_errors[0] = True
                        ret = 0
                    for i in range(count):
                        for j in range(points):
                            for k in range(ax_c):
                                if population[i][j][k] < axes[k][0] or population[i][j][k] > axes[k][1]:
                                    if self.verbosity > 0 and not first_errors[1]:
                                        print("ERROR: population function didn't output in axes range")
                                        first_errors[1] = True
                                    ret = 0
        if ret == 1 and self.verbosity > 1:
            print("Population function correctly ran in all test cases.")
        return ret

    def c_test_mutate_function(self, function):
        """
        Tests if mutate function is a valid mutate function.

        Returns:
            int: 1 if successful, 0 if not
        """
        ret = 1
        first_errors = [False, False]
        for count in range(10, 25, 5):
            for points in range(5, 10):
                for ax_c in range(3, 5):
                    axes = []
                    for _ in range(ax_c):
                        axes.append((np.random.random_sample()*2), (3+np.random.random_sample()*4))
                    population = GeneticAlgorithms.random_population(count, points, axes)  # assumes this works
                    population = function(population, axes)
                    if population.shape != (count, points, ax_c):
                        if self.verbosity > 0 and not first_errors[0]:
                            print("ERROR: mutate function didn't output valid shape")
                            first_errors[0] = True
                        ret = 0
                    for i in range(count):
                        for j in range(points):
                            for k in range(ax_c):
                                if population[i][j][k] < axes[k][0] or population[i][j][k] > axes[k][1]:
                                    if self.verbosity > 0 and not first_errors[1]:
                                        print("ERROR: mutate function didn't output in axes range")
                                        first_errors[1] = True
                                    ret = 0
        if ret == 1 and self.verbosity > 1:
            print("Mutate function correctly ran in all test cases.")
        return ret

    def c_test_cross_function(self, function):
        """
        Tests if cross function is a valid cross function

        Returns:
            int: 1 if successful, 0 if not
        """
        return general_cross_function(self.verbosity, function)

    def c_test_weighting_bias(self, function):
        """
        Tests if weighting bias function is a valid weighting bias function

        Args:
            function: weighting bias function.

        Returns:
            int: 1 if successful, 0 if not
        """
        return general_weighting_bias(self.verbosity, function)

    def c_test_set_inp(self, param, value):
        """
        Tests if inputs given to PopulationAlgorithm.set() are valid.
        Only checks that functions are callable, nothing about their attributes.

        Returns:
            int: 1 if successful, 0 if not
        """
        ret = 1
        if "__hash__" not in dir(param):  # param must be hashable
            ret = 0
            if self.verbosity > 0:
                print("ERROR: " + param + " is not hashable. It will be unable to be set in a dict.")
        else:
            if self.verbosity > 1:
                print(param + " is hashable.")
        if param in ["population_size", "time_constraint", "generations"]:
            if not ((isinstance(value, int) or
                     isinstance(value, float) or
                     isinstance(value, long))):
                if self.verbosity > 0:
                    print("ERROR: " + param + " needs to be of a number. It is " + str(value))
                ret = 0
            else:
                if self.verbosity > 1:
                    print(param + " is correctly set to a number.")
            if value < 0:
                ret = 0
                if self.verbosity > 0:
                    print("ERROR: " + param + " needs to be greater than zero.")
            else:
                if self.verbosity > 1:
                    print(param + " is greater than zero.")
            if param in ["population_size", "generations"]:
                if not isinstance(value, int):
                    ret = 0
                    if self.verbosity > 0:
                        print("ERROR: " + param + " needs to be an integer. It is " + str(value))
                else:
                    if self.verbosity > 1:
                        print(param + " is an integer.")
        if param in ["fitness_function", "population_function",
                     "mutate_function", "cross_function", "weighting_bias"]:
            if not callable(value):
                ret = 0
                if self.verbosity > 0:
                    print("ERROR: " + param + " needs to be a callable function.")
            else:
                if self.verbosity > 1:
                    print(param + " is a callable function.")
        if param == "end_condition":
            if value not in ["time_constraint", "generations"]:
                ret = 0
                if self.verbosity > 0:
                    print("ERROR: " + param + " needs to be 'time_constraint' or 'generations'")
            else:
                if self.verbosity > 1:
                    print("ERROR: " + param + " is a correct string.")
        if param == "seed":
            if not (value is None or isinstance(value, int)):
                ret = 0
                if self.verbosity > 0:
                    print("ERROR: " + param + " is incorrectly set.")
            else:
                if self.verbosity > 1:
                    print(param + " is correctly set.")
        return ret

    def c_test__cross_inp(self, weighted_population, run_locals):
        """
        Tests if inputs given to PopulationAlgorithm._cross() are valid.

        Returns:
            int: 1 if successful, 0 if not
        """
        ret = 1
        if weighted_population.shape != (run_locals["population_size"],
                                         1 + len(run_locals["axes"])):
            ret = 0
            if self.verbosity > 0:
                print("ERROR: weighted population shape doesn't match local running variables.")
        else:
            if self.verbosity > 1:
                print("Weighted population has correct shape.")
        if weighted_population[:, 0].dtype.kind not in {'b', 'u', 'i', 'f', 'c'}:
            ret = 0
            if self.verbosity > 0:
                print("ERROR: weighted population isn't filled with numbers")
        else:
            if self.verbosity > 1:
                print("Weighted population is correctly filled with numbers")
        return ret

    def c_test_run_inp(self, temp_params, base_locals):
        """
        Tests if inputs given to PopulationAlgorithm.run() are valid

        Returns:
            int: 1 if successful, 0 if not
        """
        ret = 1
        for key, val in temp_params.items():
            ret = ret and self.c_test_set_inp(key, val)
        return ret

    def random_test(self, source):
        """
        Tests if the randomness of the program is consistent with the same seed. Reproducibility

        Returns:
            int: 1 if successful, 0 if not
        """
        ret = 1
        for seed in range(1, 40):
            if source.run(temp_params={"fitness_function": (lambda x: -np.sum(x)**2+10),
                                       "population_size": 10,
                                       "time_constraint": 2,
                                       "axes": [(0, 5)],
                                       "seed": seed}) != \
                source.run(temp_params={"fitness_function": (lambda x: -np.sum(x) ** 2 + 10),
                                        "population_size": 10,
                                        "time_constraint": 2,
                                        "axes": [(0, 5)],
                                        "seed": seed}):
                ret = 0
        if ret == 0:
            if self.verbosity > 0:
                print("ERROR: Random seed non functional, results cannot be replicated.")
            return 0
        else:
            if self.verbosity > 1:
                print("Random seed functional, results replicable if a seed is used.")
            return 1


class GeneticUnitTests:
    """
    Class for unit tests on a Genetic Algorithm
    """

    def __init__(self, testing_level=1, verbosity=1):
        """
        Initializes class, does initial tests to make sure Genetic Algorithm code works, and sets continuous testing
        test level

        Args:
            testing_level (int [0, 2]): level of continuous testing to do
            level 0: no continuous testing
            level 1: only test new functions for: population, mutation, cross, and weighting <------ suggested
            level 2: test all inputs to all functions in GeneticAlgorithms
            verbosity (int [0, 3+]): verbosity level,
            level 0: no internal prints from this class
            level 1: error output only <------ suggested
            level 2: error and success outputs
            level 3+: level 2 + prints throughout running
        """
        for i in [self.test_set(),
                  self.test__cross(),
                  self.test_eval(),
                  self.test_run(),
                  self.random_test()]:
            if i != 1:
                print("WARNING: AN ERROR HAS OCCURRED IN INITIAL TESTING, THIS CLASS IS UNSTABLE.")
        self.testing_level = testing_level
        self.verbosity = verbosity

    def test_set(self):
        """
        Tests: GeneticAlgorithm.set()

        Returns:
            int: 1 if successful, 0 if not
        """
        return 1

    def test__cross(self):
        """
        Tests: GeneticAlgorithm._cross()

        Returns:
            int: 1 if successful, 0 if not
        """
        return 1

    def test_eval(self):
        """
        Tests: GeneticAlgorithm.eval()

        Returns:
            int: 1 if successful, 0 if not
        """
        return 1

    def test_run(self):
        """
        Tests: GeneticAlgorithm.run()

        Returns:
            int: 1 if successful, 0 if not
        """
        return 1

    def c_test_fitness_function(self, function):
        """
        Tests if fitness function is a valid fitness function

        Returns:
            int: 1 if successful, 0 if not
        """
        return 1

    def c_test_population_function(self, function):
        """
        Tests if population function is a valid population function

        Returns:
            int: 1 if successful, 0 if not
        """
        return 1

    def c_test_mutate_function(self, function):
        """
        Tests if mutate function is a valid mutate function

        Returns:
            int: 1 if successful, 0 if not
        """
        return 1

    def c_test_cross_function(self, function):
        """
        Tests if cross function is a valid cross function

        Returns:
            int: 1 if successful, 0 if not
        """
        return general_cross_function(self.verbosity, function)

    def c_test_weighting_bias(self, function):
        """
        Tests if weighting bias function is a valid weighting bias function

        Args:
            function: weighting bias function.

        Returns:
            int: 1 if successful, 0 if not
        """
        return general_weighting_bias(self.verbosity, function)

    def c_test_set_inp(self, param, value):
        """
        Tests if inputs given to GeneticAlgorithm.set() are valid.
        Only checks that functions are callable, nothing about their attributes.

        Returns:
            int: 1 if successful, 0 if not
        """
        ret = 1
        if "__hash__" not in dir(param):  # param must be hashable
            ret = 0
            if self.verbosity > 0:
                print("ERROR: " + param + " is not hashable. It will be unable to be set in a dict.")
        else:
            if self.verbosity > 1:
                print(param + " is hashable.")
        if param in ["population_size", "time_constraint", "generations", "point_count"]:
            if not ((isinstance(value, int) or
                     isinstance(value, float) or
                     isinstance(value, long))):
                if self.verbosity > 0:
                    print("ERROR: " + param + " needs to be of a number. It is " + str(value))
                ret = 0
            else:
                if self.verbosity > 1:
                    print(param + " is correctly set to a number.")
            if value < 0:
                ret = 0
                if self.verbosity > 0:
                    print("ERROR: " + param + " needs to be greater than zero.")
            else:
                if self.verbosity > 1:
                    print(param + " is greater than zero.")
            if param in ["population_size", "generations", "point_count"]:
                if not isinstance(value, int):
                    ret = 0
                    if self.verbosity > 0:
                        print("ERROR: " + param + " needs to be an integer. It is " + str(value))
                else:
                    if self.verbosity > 1:
                        print(param + " is an integer.")
        if param in ["fitness_function", "population_function",
                     "mutate_function", "cross_function", "weighting_bias"]:
            if not callable(value):
                ret = 0
                if self.verbosity > 0:
                    print("ERROR: " + param + " needs to be a callable function.")
            else:
                if self.verbosity > 1:
                    print(param + " is a callable function.")
        if param == "end_condition":
            if value not in ["time_constraint", "generations"]:
                ret = 0
                if self.verbosity > 0:
                    print("ERROR: " + param + " needs to be 'time_constraint' or 'generations'")
            else:
                if self.verbosity > 1:
                    print("ERROR: " + param + " is a correct string.")
        if param == "seed":
            if not (value is None or isinstance(value, int)):
                ret = 0
                if self.verbosity > 0:
                    print("ERROR: " + param + " is incorrectly set.")
            else:
                if self.verbosity > 1:
                    print(param + " is correctly set.")
        return ret

    def c_test__cross_inp(self, old_population, population_weighting, run_locals):
        """
        Tests if inputs given to GeneticAlgorithm._cross() are valid

        Returns:
            int: 1 if successful, 0 if not
        """
        return 1

    def c_test_eval_inp(self, population, run_locals):
        """
        Tests if inputs given to GeneticAlgorithm.eval() are valid

        Returns:
            int: 1 if successful, 0 if not
        """
        return 1

    def c_test_run_inp(self, temp_params, base_locals):
        """
        Tests if inputs given to GeneticAlgorithm.run() are valid

        Returns:
            int: 1 if successful, 0 if not
        """
        return 1

    def random_test(self):
        """
        Tests if the randomness of the program is consistent with the same seed. Reproducibility

        Returns:
            int: 1 if successful, 0 if not
        """
        return 1


class ParticleSwarmUnitTests:
    """
    Class for unit tests on a Particle Swarm Optimization Algorithm
    """

    def __init__(self, testing_level=1, verbosity=1):
        """
        Initializes class, does initial tests to make sure Particle Swarm Optimization Algorithm code works, and sets continuous testing
        test level

        Args:
            testing_level (int [0, 2]): level of continuous testing to do
            level 0: no continuous testing
            level 1: only test new functions for: population, mutation, cross, and weighting <------ suggested
            level 2: test all inputs to all functions in GeneticAlgorithms
            verbosity (int [0, 3+]): verbosity level,
            level 0: no internal prints from this class
            level 1: error output only <------ suggested
            level 2: error and success outputs
            level 3+: level 2 + prints throughout running
        """
        for i in [self.test_set(),
                  self.test_step(),
                  self.test_run(),
                  self.random_test()]:
            if i != 1:
                print("WARNING: AN ERROR HAS OCCURRED IN INITIAL TESTING, THIS CLASS IS UNSTABLE.")
        self.testing_level = testing_level
        self.verbosity = verbosity

    def test_set(self):
        """
        Tests: ParticleSwarmOptimization.set()

        Returns:
            int: 1 if successful, 0 if not
        """
        return 1

    def test_step(self):
        """
        Tests: ParticleSwarmOptimization.step()

        Returns:
            int: 1 if successful, 0 if not
        """
        return 1

    def test_run(self):
        """
        Tests: ParticleSwarmOptimization.run()

        Returns:
            int: 1 if successful, 0 if not
        """
        return 1

    def c_test_fitness_function(self, function):
        """
        Tests if fitness function is a valid fitness function

        Returns:
            int: 1 if successful, 0 if not
        """
        return 1

    def c_test_weighting_bias(self, function):
        """
        Tests if weighting bias function is a valid weighting bias function

        Args:
            function: weighting bias function.

        Returns:
            int: 1 if successful, 0 if not
        """
        return general_weighting_bias(self.verbosity, function)

    def c_test_set_inp(self, param, value):
        """
        Tests if inputs given to ParticleSwarmOptimization.set() are valid

        Returns:
            int: 1 if successful, 0 if not
        """
        ret = 1
        if "__hash__" not in dir(param):  # param must be hashable
            ret = 0
            if self.verbosity > 0:
                print("ERROR: " + param + " is not hashable. It will be unable to be set in a dict.")
        else:
            if self.verbosity > 1:
                print(param + " is hashable.")
        if param in ["population_size", "time_constraint", "generations", "point_count",
                     "PSO_VELOCITY_WEIGHT", "PSO_INDIVIDUAL_WEIGHT", "PSO_GROUP_WEIGHT"]:
            if not ((isinstance(value, int) or
                     isinstance(value, float) or
                     isinstance(value, long))):
                if self.verbosity > 0:
                    print("ERROR: " + param + " needs to be of a number. It is " + str(value))
                ret = 0
            else:
                if self.verbosity > 1:
                    print(param + " is correctly set to a number.")
            if value < 0:
                ret = 0
                if self.verbosity > 0:
                    print("ERROR: " + param + " needs to be greater than zero.")
            else:
                if self.verbosity > 1:
                    print(param + " is greater than zero.")
            if param in ["population_size", "generations", "point_count"]:
                if not isinstance(value, int):
                    ret = 0
                    if self.verbosity > 0:
                        print("ERROR: " + param + " needs to be an integer. It is " + str(value))
                else:
                    if self.verbosity > 1:
                        print(param + " is an integer.")
        if param in ["fitness_function", "weighting_bias"]:
            if not callable(value):
                ret = 0
                if self.verbosity > 0:
                    print("ERROR: " + param + " needs to be a callable function.")
            else:
                if self.verbosity > 1:
                    print(param + " is a callable function.")
        if param == "end_condition":
            if value not in ["time_constraint", "generations"]:
                ret = 0
                if self.verbosity > 0:
                    print("ERROR: " + param + " needs to be 'time_constraint' or 'generations'")
            else:
                if self.verbosity > 1:
                    print("ERROR: " + param + " is a correct string.")
        if param == "seed":
            if not (value is None or isinstance(value, int)):
                ret = 0
                if self.verbosity > 0:
                    print("ERROR: " + param + " is incorrectly set.")
            else:
                if self.verbosity > 1:
                    print(param + " is correctly set.")
        return ret

    def c_test_step_inp(self, particles, best_state, best_fitness, run_locals):
        """
        Tests if inputs given to ParticleSwarmOptimization.step() are valid

        Returns:
            int: 1 if successful, 0 if not
        """
        return 1

    def c_test_run_inp(self, temp_params, base_locals):
        """
        Tests if inputs given to ParticleSwarmOptimization.run() are valid

        Returns:
            int: 1 if successful, 0 if not
        """
        return 1

    def random_test(self):
        """
        Tests if the randomness of the program is consistent with the same seed. Reproducibility

        Returns:
            int: 1 if successful, 0 if not
        """
        return 1