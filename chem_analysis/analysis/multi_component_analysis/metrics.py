from typing import Callable

import numpy as np
import scipy

MetricType = Callable[[np.ndarray, np.ndarray], float]


def mean_absolute_error(D_actual: np.ndarray, D_calculated: np.ndarray) -> float:
    """ Mean absolute error """
    return np.abs(D_actual - D_calculated).sum()/D_actual.size


def mean_square_error(D_actual: np.ndarray, D_calculated: np.ndarray) -> float:
    """ Mean square error """
    return ((D_actual - D_calculated)**2).sum()/D_actual.size


def wassersein_distance_1D(D_actual: np.ndarray, D_calculated: np.ndarray, axis: int = 0) -> float:
    """
    This distance is also known as the earth mover’s distance, since it can be seen as the minimum amount
    of “work” required to transform into, where “work” is measured as the amount of distribution weight
    that must be moved, multiplied by the distance it has to be moved.

    """
    total_distance = 0
    if axis == 1:
        for col in range(D_actual.shape[1]):
            total_distance += scipy.stats.wasserstein_distance(D_actual[:, col], D_calculated[:, col])
        return total_distance

    if axis == 0:
        for row in range(D_actual.shape[0]):
            total_distance += scipy.stats.wasserstein_distance(D_actual[row, :], D_calculated[row, :])
        return total_distance

    raise ValueError("Invalid value for 'axis'.")
