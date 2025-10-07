

import numpy as np
from scipy.integrate import simpson

from chem_analysis.utils.math import get_slice
from chem_analysis.base_obj.signal_ import Signal
from chem_analysis.base_obj.signal_array import SignalArray


def integrate(signal: Signal | SignalArray, x_range: tuple[float, float]) -> float | np.ndarray:
    slice_ = get_slice(signal.x, x_range[0], x_range[1])
    if isinstance(signal, Signal):
        return np.trapz(x=signal.x[slice_], y=signal.y[slice_])

    return np.trapz(x=signal.x[slice_], y=signal.data[:, slice_], axis=1)


def integrate_simpson(signal: Signal | SignalArray, x_range: tuple[float, float]) -> float | np.ndarray:
    slice_ = get_slice(signal.x, x_range[0], x_range[1])
    if isinstance(signal, Signal):
        return simpson(x=signal.x[slice_], y=signal.y[slice_])

    return simpson(x=signal.x[slice_], y=signal.data[:, slice_], axis=1)
