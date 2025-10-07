import abc

import numpy as np
from scipy.fft import fft
from scipy.fftpack import fftshift

from chem_analysis.processing.base import ProcessingMethod


class FourierTransform(ProcessingMethod, abc.ABC):

    @abc.abstractmethod
    def run(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ...


class FastFourierTransform(FourierTransform):
    def __init__(self, degree: int = 1):
        self.degree = degree
        self._y_baseline = None

    @property
    def y_baseline(self) -> np.ndarray:
        return self._y_baseline

    def run(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return x, fftshift(fft(y))


"""
3.4.3.3. Zero-filling
Before the FID undergoes Fourier transformation into the frequency domain, it is composed of time-domain data points which define the signals. Zero-filling is a post-acquisition manipulation that adds a set of data points (zeroes) to the end of the original FID, where the signal has normally already decayed close to zero, so that the signals in the Fourier transformed spectrum are better represented by more data points. Adding the same number of zeroes as there are experimental points to the end of the FID before transformation both doubles the amount of information in the resultant spectrum and improves digital resolution. Further zero-filling just interpolates between data points, but may still aid interpretation.
"""