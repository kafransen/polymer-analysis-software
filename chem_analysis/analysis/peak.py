import abc
from collections import OrderedDict
import dataclasses
import numpy as np

import chem_analysis.utils.math as general_math
from chem_analysis.utils.printing_tables import StatsTable, apply_sig_figs


class Peak(abc.ABC):
    def __init__(self, id_: int = None):
        self.id_ = id_
        self.stats = PeakStats(self)

    @property
    @abc.abstractmethod
    def x(self) -> np.ndarray:
        ...

    @property
    @abc.abstractmethod
    def y(self) -> np.ndarray:
        ...


@dataclasses.dataclass
class PeakParent:
    x: np.ndarray
    y: np.ndarray


class PeakBounded(Peak):
    def __init__(self, parent: PeakParent, bounds: slice, id_: int = None):
        super().__init__(id_)
        self.parent = parent
        self.bounds = bounds

    def __repr__(self):
        return f"peak: {self.id_} at {self.bounds}"

    @property
    def x(self) -> np.ndarray:
        return self.parent.x[self.bounds]

    @property
    def y(self) -> np.ndarray:
        return self.parent.y[self.bounds]

    @property
    def low_bound_value(self) -> float:
        return self.parent.y[self.bounds.start]

    @property
    def high_bound_value(self) -> float:
        return self.parent.y[self.bounds.stop]

    @property
    def low_bound_location(self) -> float:
        return self.parent.x[self.bounds.start]

    @property
    def high_bound_location(self) -> float:
        return self.parent.x[self.bounds.stop]

    def get_stats(self) -> OrderedDict:
        dict_ = OrderedDict()
        dict_['slice'] = f"[{self.bounds.start}-{self.bounds.stop}]"
        dict_['slice_loc'] = f"[{apply_sig_figs(self.low_bound_location)}-{apply_sig_figs(self.high_bound_location)}]"

        return dict_

    def stats_table(self) -> StatsTable:
        return StatsTable.from_dict(self.get_stats())


class PeakStats:
    """
    area: float
        area under the peak
    mean: float
        average value
    std: float
        standard deviation
    skew: float
        skew
        symmetric: -0.5 to 0.5; moderate skew: -1 to -0.5 or 0.5 to 1; high skew: <-1 or >1;
        positive tailing to higher numbers; negative tailing to smaller numbers
    kurtosis: float
        kurtosis (Fisher) (Warning: highly sensitive to peak bounds)
        negative: flatter peak; positive: sharp peak
    full_width_half_max: float
        full width at half maximum
    asymmetry_factor: float
        asymmetry factor; distance from the center line of the peak to the back slope divided by the distance from the
        center line of the peak to the front slope;
        >1 tailing to larger values; <1 tailing to smaller numbers
    """
    def __init__(self, parent: Peak):
        self.parent = parent
        self._y_norm = None

    def _get_y_norm(self) -> np.ndarray:
        if self._y_norm is None:
            self._y_norm = self.parent.y/np.trapz(x=self.parent.x, y=self.parent.y)

        return self._y_norm

    # @property
    # def min_value(self) -> float:
    #     return np.min(self.parent.y)
    #
    # @property
    # def min_index(self) -> int:
    #     return int(np.argmin(self.parent.y))
    #
    # @property
    # def min_location(self) -> float:
    #     return self.parent.x[self.min_index]

    @property
    def max_loc(self) -> float:
        return self.parent.x[int(np.argmax(self.parent.y))]

    @property
    def max_value(self) -> float:
        return np.max(self.parent.y)

    @property
    def mean(self) -> float:
        return general_math.get_mean_of_pdf(self.parent.x, y_norm=self._get_y_norm())

    @property
    def std(self):
        return general_math.get_standard_deviation_of_pdf(self.parent.x, y_norm=self._get_y_norm(), mean=self.mean)

    @property
    def skew(self):
        return general_math.get_skew_of_pdf(self.parent.x, y_norm=self._get_y_norm(), mean=self.mean,
                                            standard_deviation=self.std)

    @property
    def kurtosis(self):
        return general_math.get_kurtosis_of_pdf(self.parent.x, y_norm=self._get_y_norm(), mean=self.mean,
                                                standard_deviation=self.std)

    @property
    def fwhm(self):
        """full_width_half_max"""
        return general_math.get_full_width_at_height(x=self.parent.x, y=self.parent.y, height=0.5)

    @property
    def asym(self):
        """asymmetry_factor"""
        return general_math.get_asymmetry_factor(x=self.parent.x, y=self.parent.y, height=0.1)

    @property
    def area(self, x: np.ndarray = None) -> float:
        if x is None:
            x = self.parent.x
        return np.trapz(x=x, y=self.parent.y)

    def get_stats(self) -> OrderedDict:
        dict_ = OrderedDict()
        properties = [attr for attr in self.__dir__() if not attr.startswith("_")]
        properties.remove("parent")
        properties.remove("get_stats")
        properties.remove("stats_table")
        for stat in properties:
            dict_[stat] = getattr(self, stat)
        return dict_

    def stats_table(self) -> StatsTable:
        return StatsTable.from_dict(self.get_stats())
