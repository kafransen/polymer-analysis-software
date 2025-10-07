from __future__ import annotations

import abc
from typing import Sequence

import numpy as np
from scipy.special import voigt_profile

from chem_analysis.analysis.peak import Peak


class PeakModel(abc.ABC):
    _args = None

    def __init__(self):
        ...

    @abc.abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        ...

    @property
    def number_args(self) -> int:
        return len(self.__slots__)

    def get_args(self) -> tuple:
        return tuple(getattr(self, arg) for arg in self._args)

    def get_kwargs(self) -> dict:
        return {arg: getattr(self, arg) for arg in self._args}

    def set_args(self, args: Sequence):
        for i, arg in enumerate(args):
            setattr(self, self._args[i], arg)

    def set_kwargs(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def get_bounds(self) -> list[tuple[float, float]]:
        bounds = []
        for k in self._args:
            bounds.append(getattr(self, k + "_bounds"))
        return bounds


class DistributionNormal(PeakModel):
    _args = ("scale", "mean", "sigma")

    def __init__(self,
                 scale: int | float = 1,
                 mean: int | float = 0,
                 sigma: int | float = 1,
                 ):
        super().__init__()
        self.scale = scale
        self.mean = mean
        self.sigma = sigma

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.scale / (self.sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - self.mean) ** 2 / (2 * self.sigma ** 2))

    def convert_to_peak(self, x: np.ndarray) -> DistributionNormalPeak:
        return DistributionNormalPeak(x, self.scale, self.mean, self.sigma)


class DistributionNormalPeak(DistributionNormal, Peak):
    def __init__(self,
                 x: np.ndarray,
                 scale: int | float = 1,
                 mean: int | float = 0,
                 sigma: int | float = 1,
                 scale_bounds: tuple[float, float] = (0, np.inf),
                 mean_bounds: tuple[float, float] = (np.inf, np.inf),
                 sigma_bounds: tuple[float, float] = (0, np.inf),
                 id_: int = None
                 ):
        DistributionNormal.__init__(self, scale, mean, sigma)
        Peak.__init__(self, id_)
        self._x = x
        self.scale_bounds = scale_bounds
        self.mean_bounds = mean_bounds
        self.sigma_bounds = sigma_bounds

    @property
    def y(self) -> np.ndarray:
        return self(self.x)

    @property
    def x(self) -> np.ndarray:
        return self._x



# from scipy.stats import cauchy
class DistributionCauchy(PeakModel):
    _args = ("scale", "mean", "gamma")

    def __init__(self,
                 scale: int | float = 1,
                 mean: int | float = 0,
                 gamma: int | float = 1,
                 ):
        super().__init__()
        self.scale = scale
        self.mean = mean
        self.gamma = gamma

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.scale / (np.pi * self.gamma * (1 + ((x - self.mean) / 2) ** 2))


class DistributionCauchyPeak(DistributionCauchy, Peak):
    def __init__(self,
                 x: np.ndarray,
                 scale: int | float = 1,
                 mean: int | float = 0,
                 gamma: int | float = 1,
                 scale_bounds: tuple[float, float] = (0, np.inf),
                 mean_bounds: tuple[float, float] = (np.inf, np.inf),
                 gamma_bounds: tuple[float, float] = (0, np.inf),
                 id_: int = None
                 ):
        DistributionCauchy.__init__(self, scale, mean, gamma)
        Peak.__init__(self, id_)
        self._x = x
        self.scale_bounds = scale_bounds
        self.mean_bounds = mean_bounds
        self.gamma_bounds = gamma_bounds

    @property
    def y(self) -> np.ndarray:
        return self(self.x)

    @property
    def x(self) -> np.ndarray:
        return self._x


class DistributionVoigt(PeakModel):
    def __init__(self,
                 scale: int | float = 1,
                 mean: int | float = 0,
                 sigma: int | float = 1,
                 gamma: int | float = 1
                 ):
        """
        gamma = 0 normal
        sigma = 0 cauchy distribution
        """
        super().__init__()
        self.scale = scale
        self.mean = mean
        self.sigma = sigma
        self.gamma = gamma

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.scale * voigt_profile(x - self.mean, sigma=self.sigma, gamma=self.gamma)


class DistributionVoigtPeak(DistributionVoigt, Peak):
    _args = ("scale", "mean", "sigma", "gamma")

    def __init__(self,
                 x: np.ndarray,
                 scale: int | float = 1,
                 mean: int | float = 0,
                 sigma: int | float = 1,
                 gamma: int | float = 1,
                 scale_bounds: tuple[float, float] = (0, np.inf),
                 mean_bounds: tuple[float, float] = (np.inf, np.inf),
                 sigma_bounds: tuple[float, float] = (0, np.inf),
                 gamma_bounds: tuple[float, float] = (0, np.inf),
                 id_: int = None
                 ):
        DistributionVoigt.__init__(self, scale, mean, sigma, gamma)
        Peak.__init__(self, id_)
        self._x = x
        self.scale_bounds = scale_bounds
        self.mean_bounds = mean_bounds
        self.gamma_bounds = gamma_bounds
        self.sigma_bounds = sigma_bounds

    @property
    def y(self) -> np.ndarray:
        return self(self.x)

    @property
    def x(self) -> np.ndarray:
        return self._x
