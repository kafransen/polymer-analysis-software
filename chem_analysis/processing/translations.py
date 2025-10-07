import abc
from typing import Sequence

import numpy as np

from chem_analysis.processing.base import ProcessingMethod
from chem_analysis.utils.math import get_slice


class Translations(ProcessingMethod, abc.ABC):
    ...


class Horizontal(Translations):
    def __init__(self,
                 shift_x: float | Sequence[float] = None,
                 shift_index: int | Sequence[int] = None,
                 wrap: bool = True,
                 ):
        if (shift_index is None) == (shift_x is None):
            raise ValueError("Provide only one: shift_x or shift_index")
        self.shift_x = shift_x
        self.shift_index = shift_index
        self.wrap = wrap

    def _get_shift_index(self, x: np.ndarray):
        if self.shift_index is not None:
            return

        if isinstance(self.shift_x, Sequence):
            self.shift_index = np.zeros_like(self.shift_x, dtype=np.int32)
            for i in range(len(self.shift_x)):
                self.shift_index[i] = self._get_shift_index_single(x, self.shift_x[i])
        else:
            self.shift_index = self._get_shift_index_single(x, self.shift_x)

    @staticmethod
    def _get_shift_index_single(x: np.ndarray, shift_x: float) -> int:
        if abs(shift_x) > (np.max(x) - np.min(x)):
            raise ValueError("shift is larger than x range")
        if shift_x > 0:
            new_min = np.min(x) + shift_x
            index = np.argmin(np.abs(x - new_min))
        else:
            new_max = np.max(x) + shift_x
            index = np.argmin(np.abs(x - new_max)) - len(x)
        return index

    def run(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(self.shift_index, Sequence) or isinstance(self.shift_x, Sequence):
            raise ValueError("For x-y signals, provide single values for 'shift_index' and 'shift_x'.")
        self._get_shift_index(x)

        if self.shift_index == 0:
            return x, y

        if self.wrap:
            y = np.roll(y, self.shift_index)
            return x, y

        if self.shift_index > 0:
            y = y[:-self.shift_index]
            x = x[self.shift_index:]
        else:
            y = y[self.shift_index:]
            x = x[:-self.shift_index]
        return x, y

    def run_array(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._get_shift_index(x)

        if isinstance(self.shift_index, int):
            if self.shift_index == 0:
                return x, y, z
            if self.wrap:
                z = np.roll(z, self.shift_index, axis=1)
                return x, y, z
            if self.shift_index > 0:
                z = z[:, -self.shift_index]
                x = x[self.shift_index:]
            else:
                z = z[:, self.shift_index:]
                x = x[:-self.shift_index]
            return x, y, z

        if not self.wrap:
            raise ValueError("'wrap must be true otherwise different x-axis are needed.")
        for i in range(z.shape[0]):
            z[i, :] = np.roll(z[i, :], self.shift_index[i])

        return x, y, z


class Subtract(Translations):
    def __init__(self, y_subtract: np.ndarray, x_subtract: np.ndarray = None):
        self.y_subtract = y_subtract
        self.x_subtract = x_subtract  # TODO: add with interplation

    def run(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return x, y - self.y_subtract

    def run_array(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return x, y, z - self.y_subtract


class AlignMax(Translations):
    def __init__(self,
                 range_: tuple[float, float] = None,
                 range_index: slice = None,
                 x_value: int | float = None,
                 wrap: bool = True
                 ):
        self.range_ = range_
        self.range_index = range_index
        self.x_value = x_value
        self.wrap = wrap
        self.x_value_index = None
        self.shift_index = None

    def run(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.x_value is None:
            raise ValueError("Set the 'AlignMax.x_value' you want to align the max to.")
        if self.range_ is not None:
            self.range_index = get_slice(x, self.range_[0], self.range_[1])
        if self.range_index is None:
            self.range_index = slice(0, -1)

        # get shift index
        max_indices = np.argmax(y[self.range_index]) + self.range_index.start
        self.x_value_index = np.argmin(np.abs(x - self.x_value))
        self.shift_index = self.x_value_index - max_indices

        translation = Horizontal(shift_index=self.shift_index, wrap=self.wrap)
        return translation.run(x, y)

    def run_array(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.range_ is not None:
            self.range_index = get_slice(x, self.range_[0], self.range_[1])
        if self.range_index is None:
            self.range_index = slice(0, -1)
        if self.x_value is None:
            # use first spectra max as reference
            self.x_value_index = np.argmax(z[0, self.range_index]) + self.range_index.start
            self.x_value = x[self.x_value_index]
        else:
            self.x_value_index = np.argmin(np.abs(x - self.x_value))

        # get shift index
        max_indices = np.argmax(z[:, self.range_index], axis=1) + self.range_index.start
        self.shift_index = self.x_value_index - max_indices

        translation = Horizontal(shift_index=self.shift_index, wrap=self.wrap)
        return translation.run_array(x, y, z)


class ScaleMax(Translations):
    def __init__(self,
                 range_: tuple[float, float] = None,
                 range_index: slice = None,
                 new_max_value: int | float = None,
                 wrap: bool = True
                 ):
        self.range_ = range_
        self.range_index = range_index
        self.new_max_value = new_max_value
        self.wrap = wrap
        self.scale = None

    def run(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.new_max_value is None:
            raise ValueError("Set the 'AlignMax.x_value' you want to align the max to.")
        if self.range_ is not None:
            self.range_index = get_slice(x, self.range_[0], self.range_[1])
        if self.range_index is None:
            self.range_index = slice(0, -1)

        self.scale = self.new_max_value / np.max(y[self.range_index])
        return x, y * self.scale

    def run_array(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.range_ is not None:
            self.range_index = get_slice(x, self.range_[0], self.range_[1])
        if self.range_index is None:
            self.range_index = slice(0, -1)
        if self.new_max_value is None:
            # use first spectra max as reference
            self.new_max_value = np.max(z[0, self.range_index])

        # get shift index
        self.scale = self.new_max_value / np.max(z[:, self.range_index], axis=1)
        return x, y, z * self.scale.reshape(-1, 1)
