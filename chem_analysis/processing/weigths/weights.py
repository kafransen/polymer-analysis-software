from __future__ import annotations
import abc
from typing import Iterable, Callable, Sequence

import numpy as np

from chem_analysis.utils.math import get_slice
from chem_analysis.utils.code_for_subclassing import MixinSubClassList
import chem_analysis.processing.weigths.penalty_functions as penalty_functions


class DataWeight(MixinSubClassList, abc.ABC):
    def __init__(self, threshold: float = 0.5, normalized: bool = True):
        self.threshold = threshold
        self.normalized = normalized

    @abc.abstractmethod
    def _get_weights(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        ...

    def get_weights(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        weights = self._get_weights(x, y)
        if np.all(weights == 0):
            raise ValueError(f"All weights are zero after applying {type(self).__name__}")

        return weights

    def get_normalized_weights(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        weights = self.get_weights(x, y)
        return weights / np.max(weights)

    def get_inverted_weights(self, x: np.ndarray, y: np.ndarray, replace_zero: float = None) -> np.ndarray:
        weights = self.get_weights(x, y)

        # avoid divide by zero
        if replace_zero is None:
            replace_zero = np.min(weights[weights > 0]) * 0.9
            # 0.9 is just to make it just a bit smaller than the smallest value

        mask = weights == 0
        weights[mask] = replace_zero

        return 1 / weights

    def get_mask(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.normalized:
            weights = self.get_normalized_weights(x, y)
        else:
            weights = self.get_weights(x, y)

        return weights <= self.threshold

    # def apply_as_mask(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    #     indexes = self.get_weights(x, y)
    #     return x[indexes], y[indexes]

    def get_weights_array(self, x: np.ndarray, _: np.ndarray, z: np.ndarray) -> np.ndarray:
        mask = np.ones_like(z)
        for i in range(z.shape[0]):
            mask[i, :] = self.get_weights(x, z[i, :])

        return mask

    def get_normalized_weights_array(self, x: np.ndarray, _: np.ndarray, z: np.ndarray) -> np.ndarray:
        mask = np.ones_like(z)
        for i in range(z.shape[0]):
            mask[i, :] = self.get_normalized_weights(x, z[i, :])

        return mask

    def get_inverted_weights_array(self, x: np.ndarray, _: np.ndarray, z: np.ndarray, replace_zero: float = None) \
            -> np.ndarray:
        mask = np.ones_like(z)
        for i in range(z.shape[0]):
            mask[i, :] = self.get_inverted_weights(x, z[i, :], replace_zero)

        return mask

    def get_mask_array(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        if self.normalized:
            weights = self.get_normalized_weights_array(x, y, z)
        else:
            weights = self.get_weights_array(x, y, z)

        return weights <= self.threshold

    # def apply_as_mask_array_index(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, index: int = 0) \
    #         -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    #     if self.normalized:
    #         weights = self.get_normalized_weights(x, z[index, :])
    #     else:
    #         weights = self.get_weights(x, z[index, :])
    #
    #     indexes = weights >= self.threshold
    #     return x[indexes], y, z[:, indexes]
    #
    # def apply_as_mask_array(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) \
    #         -> tuple[list[np.ndarray], np.ndarray, list[np.ndarray]]:
    #     """
    #     Size of x and z are different so treated separately
    #
    #     Parameters
    #     ----------
    #     x
    #     y
    #     z
    #
    #     Returns
    #     -------
    #
    #     """
    #     x_list = []
    #     z_list = []
    #     for i in range(z.shape[0]):
    #         x_, z_ = self.apply_as_mask(x, z[i, :])
    #         x_list.append(x_)
    #         z_list.append(z_)
    #
    #     return x_list, y, z_list


class DataWeightChain(DataWeight):
    def __init__(self, weights: DataWeight | Iterable[DataWeight] = None):
        super().__init__()
        if weights is None:
            weights = []
        if not isinstance(weights, Iterable):
            weights = list(weights)
        self.weights = weights

    def get_weights(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        weights = np.ones_like(x)
        for weight in self.weights:
            weights *= weight.get_weights(x, y)
        return weights

    def _get_weights(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        pass


class Slices(DataWeight):
    def __init__(self,
                 slices: slice | Iterable[slice],
                 threshold: float = 0.5,
                 normalized: bool = True,
                 invert: bool = False
                 ):
        """

        Parameters
        ----------
        slices:
            slices will be set to 1
        threshold:
            below threshold --> 0
            above threshold --> 1
        normalized:

        invert:
            flips 0 --> 1 and 1 --> 0
        """
        super().__init__(threshold, normalized)
        self.slices = slices
        self.invert = invert

    def _get_weights(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if not isinstance(self.slices, Iterable):
            if self.slices is None:
                slices = []
            else:
                slices = [self.slices]
        else:
            slices = self.slices

        if len(slices) == 0:  # TODO: implement on other weights
            return np.ones_like(x, dtype=bool)

        weights = np.zeros_like(x, dtype=bool)

        for slice_ in slices:
            weights[slice_] = 1

        if self.invert:
            return np.logical_not(weights)
        return weights


class Spans(DataWeight):
    def __init__(self,
                 x_spans: Sequence[float] | Iterable[Sequence[float]] = None,  # Sequence of length 2
                 threshold: float = 0.5,
                 normalized: bool = True,
                 invert: bool = False
                 ):
        super().__init__(threshold, normalized)
        self.x_spans = x_spans
        self.invert = invert

    def _get_weights(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if not isinstance(self.x_spans[0], Iterable):
            x_spans = [self.x_spans]
        else:
            x_spans = self.x_spans

        weights = np.zeros_like(x, dtype=bool)

        for x_span in x_spans:
            slice_ = get_slice(x, x_span[0], x_span[1])
            weights[slice_] = 1

        if self.invert:
            return np.logical_not(weights)
        return weights


class MultiPoint(DataWeight):
    def __init__(self, indexes: Iterable[int], threshold: float = 0.5, normalized: bool = True):
        """

        Parameters
        ----------
        indexes:
            index where values will be kept
        """
        super().__init__(threshold, normalized)
        self.indexes = indexes

    def _get_weights(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        weights = np.zeros_like(x)
        weights[self.indexes] = 1
        return weights


class Distance(DataWeight):
    def __init__(self,
                 reference_value: float | int = 0,
                 penalty_function: Callable = penalty_functions.penalty_function_linear,
                 threshold: float = 0.5,
                 normalized: bool = True
                 ):
        super().__init__(threshold, normalized)
        self.reference_value = reference_value
        self.penalty_function = penalty_function

    def _get_weights(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.penalty_function(y-self.reference_value)


class DistanceMedian(DataWeight):
    def __init__(self,
                 penalty_function: Callable = penalty_functions.penalty_function_linear,
                 threshold: float = 0.5,
                 normalized: bool = True
                 ):
        super().__init__(threshold, normalized)
        self.penalty_function = penalty_function

    def _get_weights(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.penalty_function(y-np.median(y))


def get_distance_remove(y: np.ndarray, amount: float = 0.7, speed: float = 0.1, max_iter: int = 1000):
    stop_len = len(y) * amount
    indexes = np.ones_like(y, dtype=bool)
    for i in range(max_iter):
        dist_from_median = np.abs(y - np.median(y[indexes]))
        median_deviation = np.median(dist_from_median[indexes])
        if median_deviation == 0:
            median_deviation = np.mean(dist_from_median[indexes])
        scale_distances_from_median = dist_from_median / median_deviation
        cut_off_distance = np.max(scale_distances_from_median[indexes]) * (1 - 0.5 * np.exp(-i / speed))
        keep_points = scale_distances_from_median < cut_off_distance
        if np.sum(keep_points) == 0:
            break
        indexes = np.bitwise_and(indexes, keep_points)

        if np.sum(indexes) < stop_len:
            break

    return indexes


class AdaptiveDistanceMedian(DataWeight):
    def __init__(self,
                 penalty_function: Callable = penalty_functions.penalty_function_linear,
                 amount: float = 0.5,
                 speed: float = 10,
                 max_iter: int = 1000,
                 threshold: float = 0.5,
                 normalized: bool = True
                 ):
        super().__init__(threshold, normalized)
        self.amount = amount
        self.speed = speed
        self.max_iter = max_iter
        self.indexes = None
        self.penalty_function = penalty_function

    def _get_weights(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.indexes = get_distance_remove(y, self.amount, self.speed, self.max_iter)
        return self.penalty_function(y-np.median(y[self.indexes]))
