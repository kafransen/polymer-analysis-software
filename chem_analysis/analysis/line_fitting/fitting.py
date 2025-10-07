from typing import Sequence
from itertools import chain

import numpy as np
from scipy.optimize import curve_fit

from chem_analysis.analysis.line_fitting.peak_models import PeakModel


class PeaksMultiple:
    def __init__(self, peaks: Sequence[PeakModel]):
        self.peaks = peaks

    def __call__(self, x: np.ndarray, *args) -> np.ndarray:
        self.set_args(args)

        y = np.zeros_like(x, dtype=x.dtype)
        for i, peak in enumerate(self.peaks):
            y += peak(x)

        return y

    def get_args(self) -> tuple:
        return tuple(chain(*(peak.get_args() for peak in self.peaks)))

    def set_args(self, args):
        args = list(args)
        try:
            for peak in self.peaks:
                args_ = args[:peak.number_args]
                del args[:peak.number_args]
                peak.set_args(args_)
        except IndexError:
            raise IndexError("too few args")
        if len(args) != 0:
            raise IndexError("not all args used")

    def get_bounds(self) -> list[tuple[float, float]]:
        bounds = []
        for peak in self.peaks:
            bounds += peak.get_bounds()

        return bounds


class ResultPeakFitting:
    def __init__(self):
        self.multipeak: PeaksMultiple | None = None
        self.covariance = None

    @property
    def peaks(self) -> Sequence:
        return self.multipeak.peaks


def peak_deconvolution(
        peaks: Sequence[PeakModel],
        xdata: np.ndarray,
        ydata: np.ndarray,
        **kwargs
) -> ResultPeakFitting:
    result = ResultPeakFitting()
    multipeak = PeaksMultiple(peaks)

    output = curve_fit(
        f=multipeak,
        xdata=xdata,
        ydata=ydata,
        p0=multipeak.get_args(),
        bounds=multipeak.get_bounds(),
        **kwargs
    )
    args, covariance = output
    multipeak.set_args(args)

    result.multipeak = multipeak
    result.covariance = covariance
    return result


def peak_deconvolution_with_n_peaks(
        peaks: Sequence[type],
        xdata: np.ndarray,
        ydata: np.ndarray,
        **kwargs
) -> ResultPeakFitting:
    peaks = [peak() for peak in peaks]

    # get initial conditions


    result = ResultPeakFitting()
    multipeak = PeaksMultiple(peaks)

    output = curve_fit(
        f=multipeak,
        xdata=xdata,
        ydata=ydata,
        p0=multipeak.get_args(),
        bounds=multipeak.get_bounds(),
        **kwargs
    )
    args, covariance = output
    multipeak.set_args(args)

    result.multipeak = multipeak
    result.covariance = covariance
    return result


def peak_deconvolution_auto(
        peaks: Sequence[type],
        xdata: np.ndarray,
        ydata: np.ndarray,
        **kwargs
) -> ResultPeakFitting:
    # do magic

    # get initial conditions

    result = ResultPeakFitting()
    multipeak = PeaksMultiple(peaks)

    output = curve_fit(
        f=multipeak,
        xdata=xdata,
        ydata=ydata,
        p0=multipeak.get_args(),
        bounds=multipeak.get_bounds(),
        **kwargs
    )
    args, covariance = output
    multipeak.set_args(args)

    result.multipeak = multipeak
    result.covariance = covariance
    return result
