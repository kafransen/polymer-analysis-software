from __future__ import annotations

import abc
import copy

import numpy as np

from chem_analysis.utils.code_for_subclassing import MixinSubClassList


class ProcessingMethod(MixinSubClassList, abc.ABC):

    @abc.abstractmethod
    def run(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ...

    @abc.abstractmethod
    def run_array(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ...


class Processor:
    """
    Processor
    """
    def __init__(self, methods: list[ProcessingMethod] = None):
        self._methods: list[ProcessingMethod] = [] if methods is None else methods
        self.processed = False

    def __repr__(self):
        return f"Processor: {len(self)} methods"

    def __len__(self):
        return len(self._methods)

    @property
    def methods(self) -> list[ProcessingMethod]:
        return self._methods

    def add(self, *args: ProcessingMethod):
        self._methods += args
        self.processed = False

    def insert(self, index: int, method: ProcessingMethod):
        self._methods.insert(index, method)
        self.processed = False

    def delete(self, method: int | ProcessingMethod):
        if isinstance(method, ProcessingMethod):
            self._methods.remove(method)
        else:
            self._methods.pop(method)
        self.processed = False

    def run(self, x: np.ndarray, y: np.ndarray, z: np.ndarray | None = None) \
            -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
        for method in self._methods:
            if z is None:
                x, y = method.run(x, y)
            else:
                x, y, z = method.run_array(x, y, z)

        self.processed = True
        if z is None:
            return x, y
        return x, y, z

    def get_copy(self) -> Processor:
        copy_ = copy.deepcopy(self)
        copy_.processed = False
        return copy_
