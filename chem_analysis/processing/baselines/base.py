import abc
from typing import Iterable

import numpy as np
from scipy.optimize import minimize_scalar

from chem_analysis.processing.base import ProcessingMethod
from chem_analysis.processing.weigths.weights import DataWeight, DataWeightChain


class BaselineCorrection(ProcessingMethod, abc.ABC):
    def __init__(self, weights: DataWeight | Iterable[DataWeight] = None):
        if weights is not None and isinstance(weights, Iterable):
            weights = DataWeightChain(weights)
        self.weights: DataWeight = weights
        self._x = None
        self._y = None

    @property
    def y(self) -> np.ndarray | None:
        return self._y

    @property
    def x(self) -> np.ndarray | None:
        return self._x

    def run(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._y = self.get_baseline(x, y)
        self._x = x
        return x, y - self._y

    def run_array(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._y = self.get_baseline_array(x, y, z)
        self._x = x
        return x, y, z - self._y

    @abc.abstractmethod
    def get_baseline(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        ...

    def get_baseline_array(self, x: np.ndarray, _: np.ndarray, z: np.ndarray) -> np.ndarray:
        baseline = np.empty_like(z)

        for i in range(z.shape[0]):
            baseline[i, :] = self.get_baseline(x, z[i, :])

        return baseline


class Polynomial(BaselineCorrection):
    def __init__(self,
                 degree: int = 1,
                 poly_weights: np.ndarray = None,
                 weights: DataWeight | Iterable[DataWeight] = None
                 ):
        super().__init__(weights)
        self.degree = degree
        self.poly_weights = poly_weights

    def get_baseline(self, x: np.ndarray, y: np.ndarray, poly_weights: np.ndarray = None) -> np.ndarray:
        if poly_weights is None:
            if self.poly_weights is None:
                poly_weights = np.ones_like(y)
            else:
                poly_weights = self.poly_weights

        if self.weights is not None:
            mask = self.weights.get_mask(x, y)
            x_ = x[mask]
            y_ = y[mask]
            w_ = poly_weights[mask]
        else:
            x_ = x
            y_ = y
            w_ = poly_weights

        params = np.polyfit(x_, y_, self.degree, w=w_)
        func_baseline = np.poly1d(params)
        return func_baseline(x)

    def get_baseline_array(self, x: np.ndarray, _: np.ndarray, z: np.ndarray) -> np.ndarray:
        baseline = np.empty_like(z)

        if self.poly_weights is None:
            self.poly_weights = np.ones_like(z[0, :])

        if self.poly_weights.shape == z.shape:
            for i in range(z.shape[0]):
                baseline[i, :] = self.get_baseline(x, z[i, :], self.poly_weights[i, :])
        elif self.poly_weights.size == z.shape[1]:
            for i in range(z.shape[0]):
                baseline[i, :] = self.get_baseline(x, z[i, :], self.poly_weights)
        else:
            raise ValueError(f"{type(self).__name__}.poly_weights is wrong shape."
                             f"\n\texpected: {z.shape} or {z.shape[1]}"
                             f"\n\tgiven: {self.poly_weights.shape}")

        return baseline


class Subtract(BaselineCorrection):
    def __init__(self,
                 y: np.ndarray,
                 x: np.ndarray = None,
                 multiplier: float = 1,
                 ):
        super().__init__(None)
        self.y_sub = y
        self.x_sub = x
        self.multiplier = multiplier

    def get_baseline(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if len(self.y_sub) == len(y):
            return self.multiplier * self.y_sub

        raise NotImplementedError()  # TODO: x-interpolation

    def get_baseline_array(self, x: np.ndarray, _: np.ndarray, z: np.ndarray) -> np.ndarray:
        baseline = np.empty_like(z)

        for i in range(z.shape[0]):
            baseline[i, :] = self.get_baseline(x, z[i, :])

        return baseline


class SubtractOptimize(BaselineCorrection):
    def __init__(self,
                 y: np.ndarray,
                 x: np.ndarray = None,
                 weights: DataWeight | Iterable[DataWeight] = None,
                 bounds: tuple[float, float] = (-2, 2)
                 ):
        super().__init__(weights)
        self.y_sub = y
        self.x_sub = x
        self.bounds = bounds

        self.multiplier = None

    def get_baseline(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.weights is not None:
            mask = self.weights.get_mask(x, y)
            x_ = x[mask]
            y_ = y[mask]
            y_sub = self.y_sub[mask]
            x_sub = self.x_sub[mask]
        else:
            x_ = x
            y_ = y
            y_sub = self.y_sub
            x_sub = self.x_sub

        self.multiplier = self._get_multiplier(x_, y_, x_sub, y_sub)
        return self.multiplier * self.y_sub

    def _get_multiplier(self, x: np.ndarray, y: np.ndarray, x_sub: np.ndarray, y_sub: np.ndarray) -> float:
        if len(self.y_sub) == len(y):
            def func(m) -> float:
                return float(np.sum(np.abs(y-m*y_sub)))

            result = minimize_scalar(func, bounds=self.bounds)
            if not result.success:
                raise ValueError(f"'{type(self).__name__}' has not converged.")
            return result.x

        raise NotImplementedError()  # TODO: x-interpolation

    def get_baseline_array(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        self.multiplier = np.empty_like(y)
        baseline = np.empty_like(z)

        for i in range(z.shape[0]):
            y = z[i, :]
            if self.weights is not None:
                mask = self.weights.get_mask(x, y)
                x_ = x[mask]
                y_ = y[mask]
                y_sub = self.y_sub[mask]
                x_sub = self.x_sub[mask]
            else:
                x_ = x
                y_ = y
                y_sub = self.y_sub
                x_sub = self.x_sub

            multiplier = self._get_multiplier(x_, y_, x_sub, y_sub)
            self.multiplier[i] = multiplier
            baseline[i, :] = multiplier * self.y_sub

        return baseline

# Bernstein polynomial (order = 3)
# Splines
# multipoint
