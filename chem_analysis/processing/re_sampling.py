import abc
from typing import Sequence, Iterable
import logging

import numpy as np

from chem_analysis.processing.base import ProcessingMethod
from chem_analysis.processing.weigths.weights import Slices, Spans
from chem_analysis.utils.math import get_slice


class ReSampling(ProcessingMethod, abc.ABC):
    ...


class EveryN(ReSampling):
    def __init__(self,
                 x_step: int = None,
                 y_step: int = None,
                 start_index_x: int = 0,
                 start_index_y: int = 0
                 ):
        if x_step is None and y_step is None:
            raise ValueError(f"Both '{type(self).__name__}.x_step' and '{type(self).__name__}.y_step' can't be None.")
        if x_step is not None and (x_step < 1):
            raise ValueError(f"'{type(self).__name__}.x_step' must be 1 or greater.")
        if y_step is not None and (y_step < 1):
            raise ValueError(f"'{type(self).__name__}.y_step' must be 1 or greater.")
        self.x_step = x_step
        self.y_step = y_step
        self.start_index_x = start_index_x
        self.start_index_y = start_index_y

    def run(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.y_step is not None:
            logging.warning(f"'{type(self).__name__}.y_step' not used in Signal processing.")
        if self.x_step is None:
            raise ValueError(f"'{type(self).__name__}.x_step' needs to be defined.")
        return x[self.start_index_x::self.x_step], y[self.start_index_x::self.x_step]

    def run_array(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        y_step = self.y_step or 1
        x_step = self.x_step or 1
        return x[self.start_index_x::x_step], y[self.start_index_y::y_step], \
               z[self.start_index_y::y_step, self.start_index_x::x_step]


class CutSlices(ReSampling):
    def __init__(self,
                 x_slices: slice | Iterable[slice] = None,
                 y_slices: slice | Iterable[slice] = None,
                 invert: bool = False
                 ):
        if x_slices is None and y_slices is None:
            raise ValueError(f"Both '{type(self).__name__}.x_step' and '{type(self).__name__}.y_step' can't be None.")
        self.x_slices = x_slices
        self.y_slices = y_slices
        self.invert = invert

    def run(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.y_slices is not None:
            logging.warning(f"'{type(self).__name__}.y_slices' not used in Signal processing.")
        if self.x_slices is None:
            raise ValueError(f"'{type(self).__name__}.x_slices' needs to be defined.")
        slice_ = Slices(self.x_slices, invert=self.invert)
        mask = slice_.get_mask(x, y)
        return x[mask], y[mask]

    def run_array(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.x_slices is not None:
            slice_x = Slices(self.x_slices, invert=self.invert)
            mask_x = slice_x.get_mask(x, z)
        else:
            mask_x = np.ones_like(x, dtype=np.bool)

        if self.y_slices is not None:
            slice_y = Slices(self.y_slices, invert=self.invert)
            mask_y = slice_y.get_mask(y, z)
        else:
            mask_y = np.ones_like(y, dtype=np.bool)

        z = z[mask_y]
        return x[mask_x], y[mask_y], z[:, mask_x]


class CutSpans(ReSampling):
    def __init__(self,
                 x_spans: Sequence[float] | Iterable[Sequence[float]] = None,  # Sequence of length 2
                 y_spans: Sequence[float] | Iterable[Sequence[float]] = None,  # Sequence of length 2
                 invert: bool = False
                 ):
        if x_spans is None and y_spans is None:
            raise ValueError("Both 'EveryN.x_step' and 'EveryN.y_step' can't be None.")
        self.x_spans = x_spans
        self.y_spans = y_spans
        self.invert = invert

    def run(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.y_spans is not None:
            logging.warning(f"'{type(self).__name__}.y_spans' not used in Signal processing.")
        if self.x_spans is None:
            raise ValueError(f"'{type(self).__name__}.x_spans' needs to be defined.")
        slice_ = Spans(self.x_spans, invert=self.invert)
        mask = slice_.get_mask(x, y)
        return x[mask], y[mask]

    def run_array(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.x_spans is not None:
            slice_x = Spans(self.x_spans, invert=self.invert)
            mask_x = slice_x.get_mask(x, z)
        else:
            mask_x = np.ones_like(x, dtype=np.bool)

        if self.y_spans is not None:
            slice_y = Spans(self.y_spans, invert=self.invert)
            mask_y = slice_y.get_mask(y, z)
        else:
            mask_y = np.ones_like(y, dtype=np.bool)

        return x[mask_x], y[mask_y], z[mask_y, mask_x]


class CutOffValue(ReSampling):
    def __init__(self,
                 x_span: float | Sequence[float],
                 cut_off_value: float | int,
                 invert: bool = False
                 ):
        super().__init__()
        self.x_span = x_span
        self.cut_off_value = cut_off_value
        self.invert = invert
        self.index = None

    def run(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("Not valid method for Signals")

    def run_array(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        index = self.get_index(x, z)
        self.index = index
        return x, y[index], z[index]

    def get_index(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        if isinstance(self.x_span, Sequence):
            slice_ = get_slice(x, self.x_span[0], self.x_span[1])
            indexes = np.any(z[:, slice_] > self.cut_off_value)
        else:
            index = np.argmin(np.abs(x - self.x_span))
            indexes = z[:, index] < self.cut_off_value

        if self.invert:
            return np.logical_not(indexes)
        return indexes

# class AveragingEveryN(ProcessingMethod, abc.ABC):
#     def __init__(self, weights: DataWeight | Iterable[DataWeight] = None):
