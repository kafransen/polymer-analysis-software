import logging
from collections.abc import Sequence
from typing import Callable

import numpy as np
from scipy.optimize import brentq


def check_bounds(bound: Sequence[int | float]) -> tuple[int | float, int | float]:
    if len(bound) != 2:
        raise ValueError("Bound must have len() == 2.")
    if not (isinstance(bound[0], int) or isinstance(bound[1], float)):
        raise TypeError("Bounds must have values that are int or float.")
    if bound[0] == bound[1]:
        raise ValueError("The lower and upper bounds can not be equal.")
    if bound[0] > bound[1]:
        bound = (bound[1], bound[0])
    return tuple(bound)


def compute_x_bound_from_y_bound(func: Callable, y_bound: tuple[int | float, int | float]) \
        -> tuple[int | float, int | float] | None:
    b = 100
    for i in range(5):
        try:
            lb_x, result_lb = brentq(lambda x: func(x) - y_bound[0], 1, b, full_output=True)
        except ValueError:
            b = b*100
            continue
        if result_lb.converged:
            break
    else:
        logging.error("Could not converge Calibration.y_bound calculation. Take cautions using the calibration.")
        return None

    try:
        ub_x, result_ub = brentq(lambda x: func(x) - y_bound[1], 1, lb_x, full_output=True)
        if not result_ub.converged:
            logging.error("Could not converge Calibration.y_bound calculation. Take cautions using the calibration.")
            return None
    except ValueError:
        logging.error("Could not converge Calibration.y_bound calculation. Take cautions using the calibration.")
        return None

    return lb_x, ub_x


class Calibration:
    def __init__(self,
                 func: Callable,
                 *,
                 x_bounds: Sequence[int | float] = None,
                 y_bounds: Sequence[int | float] = None,
                 name: str = None
                 ):
        self.name = name
        self.func = func

        self._x_bounds = None
        self._y_bounds = None
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds

    def __repr__(self):
        text = ""
        if self.name is not None:
            text += self.name
        else:
            text += self.func.__name__
        if self.y_bounds is not None:
            text += f"{self.y_bounds}"
        return text

    @property
    def x_bounds(self) -> tuple[int | float, int | float] | None:
        return self._x_bounds

    @x_bounds.setter
    def x_bounds(self, x_bounds: Sequence[int | float]):
        if x_bounds is None:
            return
        self._x_bounds = check_bounds(x_bounds)
        self._y_bounds = (self.func(x_bounds[0]), self.func(x_bounds[1]))

    @property
    def y_bounds(self) -> tuple[int | float, int | float] | None:
        return self._y_bounds

    @y_bounds.setter
    def y_bounds(self, y_bounds: Sequence[int | float]):
        if y_bounds is None:
            return
        self._y_bounds = check_bounds(y_bounds)
        self._x_bounds = compute_x_bound_from_y_bound(self.func, self.y_bounds)

    def get_y(self, x: int | float | np.ndarray, with_bounds: bool = True) -> int | float | np.ndarray:
        """

        Parameters
        ----------
        x:
            values to evaluate at
        with_bounds:
            will set values outside bounds to zero

        Returns
        -------
        returns 0 if outside of bounds
        """
        y = self.func(x)

        if with_bounds:
            if isinstance(y, int) or isinstance(y, float):
                if self.y_bounds[0] < y < self.y_bounds[1]:
                    return y
                else:
                    return 0
            else:
                mask = y < self.y_bounds[0]
                y[mask] = 0
                mask = y > self.y_bounds[1]
                y[mask] = 0

        return y
