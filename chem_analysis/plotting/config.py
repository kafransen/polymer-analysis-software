import enum
import logging

from chem_analysis.config import global_config
from chem_analysis.plotting.plot_format import get_plot_color
from chem_analysis.base_obj.signal_ import Signal
from chem_analysis.base_obj.calibration import Calibration
from chem_analysis.base_obj.signal_array import SignalArray
from chem_analysis.sec import SECSignal, SECSignalArray
from chem_analysis.sec.sec_calibration import SECCalibration
from chem_analysis.nmr import NMRSignal, NMRSignalArray
from chem_analysis.ir import IRSignal, IRSignalArray
from chem_analysis.processing.baselines.base import BaselineCorrection
from chem_analysis.analysis.boundary_detection.boundary_detection import ResultPeakBound

logger = logging.getLogger("plotting")


class NormalizationOptions(enum.Enum):
    NONE = 0
    AREA = 1
    PEAK_HEIGHT = 2


class SignalColorOptions(enum.Enum):
    SINGLE = 0
    DIVERSE = 1


class PlotConfig:
    normalizations = NormalizationOptions
    signal_colors = SignalColorOptions

    def __init__(self):
        self.default_formatting: bool = True
        self.title = None
        self.y_label = None
        self.x_label = None
        self.z_label = None

        # signal
        self.normalize: NormalizationOptions = NormalizationOptions.NONE
        # self.signal_color: Callable | str = 'rgb(10,36,204)'
        self.signal_line_width = 2
        self._signal_color_counter = 0
        self._set_color_counter = None
        self._set_color_array_ = None

        # peak
        # self.peak_show_trace: bool = False
        self.peak_show_shade: bool = True
        self.peak_show_bounds: bool = False
        self.peak_show_max: bool = True
        self.signal_connect_gaps: bool = False
        self.signal_color: str | SignalColorOptions | None = None
        self.peak_bound_color: str | None = None
        self.peak_bound_line_width: float = 3
        self.peak_bound_height: float = 0.05  # % of max
        self.peak_marker_size: float = 3

    def set_attrs_from_kwargs(self, **kwargs):
        if not kwargs:
            return

        for k, v in kwargs.items():
            if not hasattr(self, k):
                logger.warning(f"'{k} is an invalid plot configuration.")
            setattr(self, k, v)

    ## layout ########################################################################################################
    def get_x_label(self, signal: Signal) -> str:
        if self.x_label is not None:
            return self.x_label
        return signal.x_label

    def get_y_label(self, signal: Signal) -> str:
        if self.y_label is not None:
            return self.y_label
        return signal.y_label

    def get_z_label(self, signal: SignalArray) -> str:
        if self.z_label is not None:
            return self.z_label
        return signal.z_label

    ## signal ########################################################################################################
    def get_signal_color(self, signal: Signal) -> str:
        if self._set_color_counter is None:
            return 'rgb(10,36,204)'

        return self._set_color_array[signal.id_]

    @property
    def _set_color_array(self):
        if self._set_color_array_ is None:
            if self.signal_color is None or self.signal_color is SignalColorOptions.SINGLE:
                self._set_color_array_ = get_plot_color(self._set_color_counter)

        return self._set_color_array_

    def set_color_count(self, number_colors: int):
        self._set_color_counter = number_colors

    def get_signal_group(self, signal: Signal) -> str:
        return f"sig_{signal.id_}"