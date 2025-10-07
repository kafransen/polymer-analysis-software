from typing import Callable

import numpy as np

penalty_function = Callable[[np.ndarray, float | int], np.ndarray]


def penalty_function_constant(x: np.ndarray, constant: float = 1) -> np.ndarray:
    penalty = np.ones_like(x)
    mask = np.abs(x) < constant
    penalty[mask] = 0
    return penalty


def penalty_function_linear(x: np.ndarray, prefactor: int = 1) -> np.ndarray:
    return prefactor * np.abs(x)


def penalty_function_quadratic(x: np.ndarray, prefactor: int = 1) -> np.ndarray:
    return prefactor * (x)**2


def penalty_function_polynomial(x: np.ndarray,  prefactor: int = 1, power: int| float = 3) \
        -> np.ndarray:
    return prefactor * np.abs(x**power)


def penalty_function_log(x: np.ndarray, prefactor: int = 1) -> np.ndarray:
    return prefactor * np.log(abs(x)+1)
