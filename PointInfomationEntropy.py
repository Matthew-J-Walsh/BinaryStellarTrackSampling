import numpy as np
import matplotlib.pyplot as plt
import math


SET_HELPER_ITERATIONS = 10


def entropy(v):
    """
    Entropy

    Args:
        v (float): input value

    returns:
        float: Information Entropy
    """
    return np.sum(-v*np.log(v))


def entropy_of_point(point, rest_points, rest_classes):
    """
    Entropy of a point given the rest of the points

    Args:
        point (np.array float): point to figure out entropy at
        rest_points (np.array float): the rest of the points
        rest_classes (np.array float): the rest of the points classes

    Returns:
        float: Information Entropy
    """
    pressure = set_pressure(point.flatten(), rest_points, rest_classes).flatten()
    pure = np.sum(pressure)
    norm = pressure / pure
    return entropy(norm)  # / pure (max(1-norm.max(),0))


def set_pressure(point, input_table, class_table):
    """
    "Entropic" pressure on point from the rest of the set

    Args:
        point (np.array float): point to figure out entropy at
        input_table (np.array float): the rest of the points in the set
        class_table (np.array float): the rest of the points in the set classes

    Returns:
        float: Pressure
    """
    return np.matmul(1/(np.square(np.sum(np.square(input_table - point).T, axis=0))), class_table)


def entropy_of_set_helper(point, rest, og_inp, og_table, memoed):
    """
    Entropy and pressure of a point given a set.

    Args:
        point (np.array float): point to figure out entropy at
        rest (np.array float): rest of the points in the set
        og_inp (np.array float): the points where the data is already taken
        og_table (np.array float): the classes of the points where the data is already taken
        memoed (dict - ids): memoization table

    Returns:
        float: Entropy and pressure of point
    """
    pressure_values = np.empty((rest.shape[0], og_table.shape[1]))
    for i in range(rest.shape[0]):
        i_d = id(rest[i])
        if i_d not in memoed.keys():
            pressure_values[i] = set_pressure(rest[i], og_inp, og_table)
            memoed.update({i_d: np.copy(pressure_values[i])})
        else:
            pressure_values[i] = np.copy(memoed[i_d])
    prob_vals = pressure_values / np.sum(pressure_values, axis=1)[:, np.newaxis]
    val = 0
    for _ in range(SET_HELPER_ITERATIONS):
        if len(prob_vals)>0:
            rd_vals = np.floor(np.random.rand(prob_vals.shape[0])*prob_vals.shape[1]).astype(int)
            rest_vals = np.zeros((rest.shape[0], og_table.shape[1]))
            rest_vals[np.arange(0, rest_vals.shape[0]), rd_vals] = 1
            val += min(entropy_of_point(point,
                                        np.concatenate((og_inp, rest), axis=0),
                                        np.concatenate((og_table, rest_vals), axis=0)),
                       entropy_of_point(point,
                                        og_inp,
                                        og_table))
        else:
            val += entropy_of_point(point,
                                    og_inp,
                                    og_table)
    return val


def entropy_of_set(points, og_inp, og_table):
    """
    Entropy and pressure of an entire set.

    Args:
        points (np.array float): points to figure out entropy and co-pressure at
        og_inp (np.array float): the points where the data is already taken
        og_table (np.array float): the classes of the points where the data is already taken

    Returns:
        float: Entropy + Pressure of the entire set
    """
    memoed = {}
    #new_points = np.concatenate((points, np.arange(points.shape[0]).T.reshape(points.shape[0], 1)), axis=1)
    adv_points = np.empty((points.shape[0], points.shape[0], points.shape[1]))
    for i in range(points.shape[0]):
        adv_points[i] = np.roll(points, i*(points.shape[1]))
    val = 0
    for i in range(adv_points.shape[0]):
        val += entropy_of_set_helper(adv_points[i][0], adv_points[i][1:], og_inp, og_table, memoed)
    return val


def get_information_entropy_function(og_inp, og_table):
    """
    Shortened Function.

    Args:
        og_inp (np.array float): the points where the data is already taken
        og_table (np.array float): the classes of the points where the data is already taken

    Returns:
        lambda: Function that takes a single input of points for a given set of
        points with data already taken at them
    """
    return lambda x: entropy_of_set(x, og_inp, og_table)


def visualize_entropy(og_inp, og_table):
    """
    Visualizes the single point entropy from a set of

    Args:
        og_inp (np.array float): the points where the data is already taken
        og_table (np.array float): the classes of the points where the data is already taken
    """
    dx, dy = .01, .01
    y, x = np.mgrid[slice(0.0, 1.0 + dy, dy),
                    slice(0.0, 1.0 + dx, dx)]

    con = np.concatenate((x[:, :, np.newaxis], y[:, :, np.newaxis]), axis=2)

    z = np.empty((con.shape[0], con.shape[1]))
    for i in range(con.shape[0]):
        for j in range(con.shape[1]):
            val = entropy_of_point(con[j, con.shape[0] - i - 1], og_inp, og_table)
            z[i, j] = val

    plt.pcolormesh(x, y, z, cmap='RdBu', vmin=0)
    return


def visualize_entropy_and_points(points, og_inp, og_table):
    """
    Visualizes the given points on the entropy graph.

    Args:
        points (np.array float): the points to visualizes (usually chosen points for iteration from optimization)
        og_inp (np.array float): the points where the data is already taken
        og_table (np.array float): the classes of the points where the data is already taken
    """
    visualize_entropy(og_inp, og_table)
    plt.scatter(points[:, 1], 1 - points[:, 0], c=['#00ff00'] * points.shape[0])
    return













