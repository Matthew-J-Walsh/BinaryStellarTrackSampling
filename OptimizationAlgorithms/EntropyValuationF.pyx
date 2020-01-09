import numpy as np


weightings = np.array([1, 1])


def entropy(v):
    return np.sum(-v * np.log(v))


def value_of_point(point, rest_points, rest_classes):
    pressure = set_pressure(point[np.newaxis, :], rest_points, rest_classes)[0]
    pure = np.sum(pressure)
    norm = pressure / pure
    return entropy(norm)


def set_pressure(point, input_table, class_table):
    inv_sq_d = 1 / (np.square(np.sum(weightings * np.square(input_table[:, np.newaxis] - point), axis=2)))
    return np.matmul(inv_sq_d.T, class_table)


def value_of_set_helper(point, rest, og_inp, og_table):
    pressure_values = np.apply_along_axis(lambda x: set_pressure(x, og_inp, og_table), 1, rest)
    pressure_values = pressure_values.reshape(pressure_values.shape[0], pressure_values.shape[2])
    prob_vals = pressure_values / np.sum(pressure_values, axis=1)[:, np.newaxis]
    val = 0
    for _ in range(1):
        if len(prob_vals) > 0:
            rd_vals = np.apply_along_axis(lambda x: np.random.choice(np.arange(0, og_table.shape[1]), p=x), 1,
                                          prob_vals)
            rest_vals = np.zeros((rest.shape[0], og_table.shape[1]))
            rest_vals[np.arange(0, rest_vals.shape[0]), rd_vals] = 1
            val += value_of_point(point,
                                  np.concatenate((og_inp, rest), axis=0),
                                  np.concatenate((og_table, rest_vals), axis=0))
        else:
            val += value_of_point(point,
                                  og_inp,
                                  og_table)
    return val


def value_of_set(points, og_inp, og_table):
    adv_points = np.empty((points.shape[0], points.shape[0], points.shape[1]))
    for i in range(points.shape[0]):
        adv_points[i] = np.roll(points, i * (points.shape[1]))
    val = 0
    for i in range(adv_points.shape[0]):
        val += value_of_set_helper(adv_points[i][0], adv_points[i][1:], og_inp, og_table)
    return val


def get_value_function(og_inp, og_table):
    return lambda x: value_of_set(x, og_inp, og_table)


