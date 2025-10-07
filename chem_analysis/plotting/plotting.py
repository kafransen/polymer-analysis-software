import logging

from chem_analysis.config import global_config
from chem_analysis.plotting.config import PlotConfig
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


def signal(
        signal_: Signal,
        *,
        fig=None,
        config: PlotConfig = None,
):
    config = config or PlotConfig()
    for option in global_config.get_plotting_options():
        if option == global_config.plotting_libraries.PLOTLY:
            if isinstance(signal, SECSignal):
                from chem_analysis.plotting.SEC_plotly import plotly_signal
                return plotly_signal(fig, signal_, config)
            if isinstance(signal, IRSignal):
                pass
            if isinstance(signal, NMRSignal):
                pass

            # default plotting
            from chem_analysis.plotting.plotting_plotly import plotly_signal
            return plotly_signal(fig, signal_, config)

        if option == global_config.plotting_libraries.MATPLOTLIB:
            pass

        if option == global_config.plotting_libraries.PYGRAPHQT:
            pass

    raise NotImplementedError()


def signal_raw(
        signal_: Signal,
        *,
        fig=None,
        config: PlotConfig = None,
):
    config = config or PlotConfig()
    for option in global_config.get_plotting_options():
        if option == global_config.plotting_libraries.PLOTLY:
            if isinstance(signal, SECSignal):
                from chem_analysis.plotting.SEC_plotly import plotly_signal_raw
                return plotly_signal_raw(fig, signal_, config)
            if isinstance(signal, IRSignal):
                pass
            if isinstance(signal, NMRSignal):
                pass

                # default plotting
            from chem_analysis.plotting.plotting_plotly import plotly_signal_raw
            return plotly_signal_raw(fig, signal_, config)

        if option == global_config.plotting_libraries.MATPLOTLIB:
            pass

        if option == global_config.plotting_libraries.PYGRAPHQT:
            pass

    raise NotImplementedError()


def peaks(
        peaks_: ResultPeakBound,
        *,
        fig=None,
        config: PlotConfig = None,
):
    config = config or PlotConfig()
    for option in global_config.get_plotting_options():
        if option == global_config.plotting_libraries.PLOTLY:
            from chem_analysis.plotting.plotting_plotly import plotly_peaks
            return plotly_peaks(fig, peaks_, config)

        if option == global_config.plotting_libraries.MATPLOTLIB:
            pass

        if option == global_config.plotting_libraries.PYGRAPHQT:
            pass

    raise NotImplementedError()


def calibration(
        calibration_: Calibration,
        *,
        fig=None,
        config=PlotConfig()
):
    config = config or PlotConfig()
    for option in global_config.get_plotting_options():
        if option == global_config.plotting_libraries.PLOTLY:
            if isinstance(calibration_, SECCalibration):
                from chem_analysis.plotting.SEC_plotly import plotly_sec_calibration
                return plotly_sec_calibration(calibration_, fig=fig, config=config)

        if option == global_config.plotting_libraries.MATPLOTLIB:
            pass

        if option == global_config.plotting_libraries.PYGRAPHQT:
            pass

    raise NotImplementedError()


def baseline(
        signal_: Signal,
        *,
        fig=None,
        config=PlotConfig()
):
    config = config or PlotConfig()
    if not signal_.processor.processed:
        _ = signal_.x  # triggers processing

    for method in signal_.processor.methods:
        if isinstance(method, BaselineCorrection):
            baseline_ = method
            break
    else:
        error_text = "No baseline correction found to add to plot."
        if fig:
            logger.warning(error_text)
            return fig
        else:
            raise RuntimeError(error_text)

    config = config or PlotConfig()
    for option in global_config.get_plotting_options():
        if option == global_config.plotting_libraries.PLOTLY:
            from chem_analysis.plotting.plotting_plotly import plotly_baseline
            return plotly_baseline(fig, baseline_, config)

        if option == global_config.plotting_libraries.MATPLOTLIB:
            pass

        if option == global_config.plotting_libraries.PYGRAPHQT:
            pass

    raise NotImplementedError()


def array_dynamic(
        array_: SignalArray,
        *,
        config=PlotConfig()
                  ):
    config = config or PlotConfig()
    for option in global_config.get_plotting_options():
        if isinstance(calibration, SECCalibration):
            pass

        if option == global_config.plotting_libraries.MATPLOTLIB:
            pass

        if option == global_config.plotting_libraries.PYGRAPHQT:
            from chem_analysis.plotting.qt_array import qt_array
            return qt_array(array_)

    raise NotImplementedError()


# def plot_signal_array_overlap(
#         array: SignalArray,
#         *,
#         config: PlotConfig = None,
#         **kwargs
# ):
#     if array.number_of_signals == 0:
#         raise ValueError("No signals to plot.")
#
#     _default_config = kwargs.pop("_default_config")
#     if config is None:
#         config = _default_config()
#         config.signal_color = SignalColorOptions.SINGLE
#     config.set_color_count(array.number_of_signals)
#     config.set_attrs_from_kwargs(**kwargs)
#
#     if global_config.plotting_library == global_config.plotting_libraries.PLOTLY:
#         return plotly_signal_array(array, config)
#
#     raise NotImplementedError()
#
#
# def plot_signal_array_3D(
#         array: SignalArray,
#         *,
#         config: PlotConfig = None,
#         **kwargs
# ):
#     if len(array.signals) == 0:
#         raise ValueError("No signals to plot.")
#
#     _default_config = kwargs.pop("_default_config")
#     if config is None:
#         config = _default_config()
#         config.signal_color = SignalColorOptions.SINGLE
#     config.set_color_count(array.number_of_signals)
#     config.set_attrs_from_kwargs(**kwargs)
#
#     if global_config.plotting_library == global_config.plotting_libraries.PLOTLY:
#         return plotly_signal_array_3D(array, config)
#
#     raise NotImplementedError()
#
#
# def plot_signal_array_surface(
#         array: SignalArray,
#         *,
#         config: PlotConfig = None,
#         **kwargs
# ):
#     if len(array.signals) == 0:
#         raise ValueError("No signals to plot.")
#
#     _default_config = kwargs.pop("_default_config")
#     if config is None:
#         config = _default_config()
#         config.signal_color = SignalColorOptions.SINGLE
#     config.set_color_count(array.number_of_signals)
#     config.set_attrs_from_kwargs(**kwargs)
#
#     if global_config.plotting_library == global_config.plotting_libraries.PLOTLY:
#         return plotly_signal_array_surface(array, config)
#
#     raise NotImplementedError()
#
#
# def plot_chromatogram(
#         chromatogram: Chromatogram,
#         *,
#         config: PlotConfig = None,
#         **kwargs
# ):
#     if len(chromatogram.signals) == 0:
#         raise ValueError("No signals to plot.")
#
#     _default_config = kwargs.pop("_default_config")
#     if config is None:
#         config = _default_config()
#         config.signal_color = SignalColorOptions.DIVERSE
#     config.set_color_count(chromatogram.number_of_signals)
#     config.set_attrs_from_kwargs(**kwargs)
#
#     if global_config.plotting_library == global_config.plotting_libraries.PLOTLY:
#         return plotly_chromatogram(chromatogram, config)
#
#     raise NotImplementedError()
