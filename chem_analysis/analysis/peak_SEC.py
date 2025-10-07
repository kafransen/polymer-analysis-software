import dataclasses
from collections import OrderedDict
import logging

import numpy as np

import chem_analysis.utils.math as general_math
from chem_analysis.sec.sec_math_functions import calculate_Mn_D_from_wi
from chem_analysis.analysis.peak import PeakBounded, PeakStats, PeakParent
from chem_analysis.base_obj.calibration import Calibration
from chem_analysis.utils.printing_tables import StatsTable

logger = logging.getLogger("chem_analysis.peak_SEC")


@dataclasses.dataclass
class PeakParentSEC(PeakParent):
    mw_i: np.ndarray | None
    calibration: Calibration | None


class PeakSEC(PeakBounded):
    """

    Attributes
    ----------
    mw_i: np.ndarray
        molecular weight of i-mer
    w_i: np.ndarray
        weight fraction of i-mer
    x_i: np.ndarray
        mole fraction of i-mer
    mw_n: float
        number average molecular weight
    mw_w: float
        weight average molecular weight
    mw_d: float
        dispersity of molecular weight
    """

    def __init__(self, parent: PeakParentSEC, bounds: slice, id_: int = None):
        super().__init__(parent, bounds, id_)
        if parent.mw_i is None:
            self.stats = PeakStats(self)
        else:
            self.stats = PeakStatsSEC(self)

        self._mw_n = None
        self._mw_d = None
        self._w_i = None
        self._x_i = None
        self._mw_i_None = False

    @property
    def mw_i(self) -> np.ndarray | None:
        if self.parent.mw_i is None:
            return None
        if self._mw_i_None:
            return None
        mw_i = self.parent.mw_i[self.bounds]
        if np.min(mw_i) == 0 and np.max(mw_i) == 0:
            # if mw_i is all zero, then peak is outside calibration and we shouldn't try to do calculations around it
            self._mw_i_None = True
            self.stats = PeakStats(self)
            return None

        return mw_i

    @property
    def w_i(self) -> np.ndarray | None:
        """ w_i = mole fraction """
        if self.mw_i is None:
            return None
        if self._w_i is None:
            self._w_i = self.y / np.abs(np.trapz(x=self.mw_i, y=self.y))  # abs because mw_i and y are reversed order
        return self._w_i

    @property
    def mw_n(self) -> float | None:
        if self.mw_i is None:
            return None
        if self._mw_n is None:
            self._mw_n, self._mw_d = calculate_Mn_D_from_wi(mw_i=self.mw_i, wi=self.w_i)
        return self._mw_n

    @property
    def mw_d(self) -> float | None:
        if self.mw_i is None:
            return None
        if self._mw_n is None:
            self._mw_n, self._mw_d = calculate_Mn_D_from_wi(mw_i=self.mw_i, wi=self.w_i)
        return self._mw_d

    @property
    def mw_w(self) -> float | None:
        if self.mw_i is None:
            return None
        return self.mw_n * self.mw_d

    @property
    def x_i(self) -> np.ndarray | None:
        """ x_i = mole fraction """
        if self.mw_i is None:
            return None
        if self._x_i is None:
            numerator = self.w_i * self.mw_n
            zero_mask = (self.mw_i == 0)
            if np.sum(zero_mask) != 0:
                logger.warning("Peak extends outside calibration window. Results may contain errors.")
            self._x_i = np.divide(numerator, self.mw_i, out=np.zeros_like(numerator), where=~zero_mask)
            # self._x_i = self.w_i * self.mw_n / self.mw_i

        return self._x_i

    def get_stats(self) -> OrderedDict:
        dict_ = super().get_stats()
        dict_["mw_n"] = self.mw_n
        dict_["mw_w"] = self.mw_w
        dict_["mw_d"] = self.mw_d
        dict_.update(self.stats.get_stats())
        return dict_

    def stats_table(self) -> StatsTable:
        return StatsTable.from_dict(self.get_stats())


class PeakStatsSEC(PeakStats):
    """

    Attributes
    ----------
    mw_max: float
        molecular weight at peak max
    mw_mean: float
        mean molecular weight distribution (same as mw_n)
    mw_std: float
        standard deviation of molecular weight distribution
    mw_skew: float
        skew of molecular weight distribution
        symmetric: -0.5 to 0.5; moderate skew: -1 to -0.5 or 0.5 to 1; high skew: <-1 or >1;
        positive tailing to higher numbers; negative tailing to smaller numbers
    mw_kurtosis: float
        kurtosis of molecular weight distribution (Fisher)  (Warning: highly sensitive to peak bounds)
        negative: flatter peak; positive: sharp peak
    mw_full_width_half_max: float
        full width half max of molecular weight distribution
    mw_asymmetry_factor: float
        asymmetry factor of molecular weight distribution; distance from the center line of the peak to the back
        slope divided by the distance from the center line of the peak to the front slope
        >1 tailing to larger values; <1 tailing to smaller numbers

    """

    def __init__(self, parent: PeakSEC):
        super().__init__(parent)

    @property
    def mw_max(self) -> float:
        return np.max(self.parent.mw_i)

    @property
    def mw_mean(self) -> float:
        return general_math.get_mean_of_pdf(self.parent.mw_i, y_norm=self.parent.x_i)

    @property
    def mw_std(self):
        return general_math.get_standard_deviation_of_pdf(self.parent.mw_i, y_norm=self.parent.x_i, mean=self.mw_mean)

    @property
    def mw_skew(self):
        return general_math.get_skew_of_pdf(self.parent.mw_i, y_norm=self.parent.x_i, mean=self.mw_mean,
                                            standard_deviation=self.mw_std)

    @property
    def mw_kurtosis(self):
        return general_math.get_kurtosis_of_pdf(self.parent.mw_i, y_norm=self.parent.x_i, mean=self.mw_mean,
                                                standard_deviation=self.mw_std)

    @property
    def mw_fwhm(self):
        """mw_full_width_half_max"""
        return general_math.get_full_width_at_height(x=self.parent.mw_i, y=self.parent.x_i, height=0.5)

    @property
    def mw_asym(self):
        """mw_asymmetry_factor"""
        return general_math.get_asymmetry_factor(x=self.parent.mw_i, y=self.parent.x_i, height=0.1)
