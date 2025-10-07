
import numpy as np


def check_array_size(x: np.ndarray, shape: tuple[int], name: str):
    if len(x.shape) != len(shape) or x.shape != shape:
        raise ValueError(f"Invalid array shape for: {name}. \n\texpected: {shape}; \n\t received: {x.shape}")


def check_array_inf_nan(x: np.ndarray, name: str):
    if np.any(np.isinf(x)):
        raise ValueError(f"The array '{name}' contains 'inf' values.")

    if np.any(np.isnan(x)):
        raise ValueError(f"The array '{name}' contains 'nan' values.")
