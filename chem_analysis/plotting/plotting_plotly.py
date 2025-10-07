
import numpy as np
import plotly.graph_objs as go

from chem_analysis.plotting.plot_format import bold_in_html
from chem_analysis.base_obj.signal_ import Signal
# from chem_analysis.base_obj.chromatogram import Chromatogram
# from chem_analysis.base_obj.signal_array import SignalArray
from chem_analysis.plotting.config import PlotConfig, NormalizationOptions
from chem_analysis.processing.baselines.base import BaselineCorrection
from chem_analysis.analysis.peak import PeakBounded
from chem_analysis.analysis.boundary_detection.boundary_detection import ResultPeakBound


def plotly_layout(fig: go.Figure, config: PlotConfig):
    if config.default_formatting:
        layout = {
            # "autosize": False,
            # "width": 1200,
            # "height": 600,
            "font": dict(family="Arial", size=18, color="black"),
            "plot_bgcolor": "white",
            # "legend": {"x": 0.05, "y": 0.95}
        }

        layout_xaxis = {
            "tickprefix": "<b>",
            "ticksuffix": "</b>",
            "showline": True,
            "linewidth": 5,
            # "mirror": True,
            "linecolor": 'black',
            "ticks": "outside",
            "tickwidth": 4,
            "showgrid": False,
            "gridwidth": 1,
            "gridcolor": 'lightgray'
        }

        layout_yaxis = {
            "tickprefix": "<b>",
            "ticksuffix": "</b>",
            "showline": True,
            "linewidth": 5,
            "mirror": True,
            "linecolor": 'black',
            "ticks": "outside",
            "tickwidth": 4,
            "showgrid": False,
            "gridwidth": 1,
            "gridcolor": 'lightgray'
        }
        fig.update_layout(**layout)
        fig.update_xaxes(**layout_xaxis)
        fig.update_yaxes(**layout_yaxis)


def plotly_signal(fig: go.Figure | None, signal: Signal, config: PlotConfig) -> go.Figure:
    if fig is None:
        fig = go.Figure()

    if config.normalize is NormalizationOptions.PEAK_HEIGHT:
        x = signal.x
        y = signal.y_normalized_by_max()
    else:
        x, y = signal.x, signal.y

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name=signal.name,
            connectgaps=config.signal_connect_gaps,
            line={"color": config.get_signal_color(signal), "width": config.signal_line_width}
        )
    )

    plotly_layout(fig, config)
    # label axis
    fig.layout.xaxis.title = bold_in_html(config.get_x_label(signal))
    fig.layout.yaxis.title = bold_in_html(config.get_y_label(signal))

    return fig


def plotly_signal_raw(fig: go.Figure | None, signal: Signal, config: PlotConfig) -> go.Figure:
    if fig is None:
        fig = go.Figure()

    if config.normalize is NormalizationOptions.PEAK_HEIGHT:
        x = signal.x_raw
        y = signal.y_raw/np.max(signal.y_raw)
    else:
        x, y = signal.x_raw, signal.y_raw

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name=signal.name + "_raw",
            connectgaps=config.signal_connect_gaps,
            line={"color": config.get_signal_color(signal), "width": config.signal_line_width}
        )
    )

    plotly_layout(fig, config)
    # label axis
    fig.layout.xaxis.title = bold_in_html(config.get_x_label(signal))
    fig.layout.yaxis.title = bold_in_html(config.get_y_label(signal))

    return fig


def plotly_peaks(fig: go.Figure, peaks: ResultPeakBound, config: PlotConfig) -> go.Figure:
    if fig is None:
        fig = go.Figure()

    for peak in peaks.peaks:
        label = f"peak {peak.id_}"
        if config.peak_show_shade:
            plotly_add_peak_shade(fig, peak, config, label)
        # if config.peak_show_trace:
        #     plotly_add_peak_trace()
        if config.peak_show_bounds:
            plotly_add_peak_bounds(fig, peak, config, label)
        if config.peak_show_max:
            plotly_add_peak_max(fig, peak, config, label)

    return fig


def plotly_add_peak_shade(fig: go.Figure, peak: PeakBounded, config: PlotConfig, label: str):
    """ Plots the shaded area for the peak. """
    fig.add_trace(go.Scatter(
        x=peak.x,
        y=peak.y,
        mode="lines",
        fill='tozeroy',
        line={"width": 0},
        showlegend=True,
        legendgroup=label,
        name=label
    ))


def plotly_add_peak_max(fig: go.Figure, peak: PeakBounded, config: PlotConfig, label: str):
    """ Plots peak name at max. """
    fig.add_trace(go.Scatter(
        x=[peak.stats.max_loc],
        y=[peak.stats.max_value],
        mode="text",
        marker={"size": config.peak_marker_size},
        text=[f"{peak.id_}"],
        textposition="top center",
        showlegend=False,
        legendgroup=label
    ))


def plotly_add_peak_bounds(fig: go.Figure, peak: PeakBounded, config: PlotConfig, label: str):
    """ Adds bounds at the bottom of the plot_add_on for peak area. """
    if config.normalize == config.normalizations.AREA:
        bound_height = np.max(peak.stats.max_value) * config.peak_bound_height
    elif config.normalize == config.normalizations.PEAK_HEIGHT:
        bound_height = config.peak_bound_height
    else:
        bound_height = np.max(peak.stats.max_value) * config.peak_bound_height

    # bounds
    fig.add_trace(go.Scatter(
        x=[peak.low_bound_location, peak.low_bound_location],
        y=[-bound_height / 2, bound_height / 2],
        mode="lines",
        line={"width": config.peak_bound_line_width, "color": 'rgb(0,0,0)'},
        showlegend=False,
        legendgroup=label
    ))
    fig.add_trace(go.Scatter(
        x=[peak.high_bound_location, peak.high_bound_location],
        y=[-bound_height / 2, bound_height / 2],
        mode="lines",
        line={"width": config.peak_bound_line_width, "color": 'rgb(0,0,0)'},
        showlegend=False,
        legendgroup=label
    ))
    fig.add_trace(go.Scatter(
        x=[peak.low_bound_location, peak.high_bound_location],
        y=[0, 0],
        mode="lines",
        line={"width": config.peak_bound_line_width, "color": 'rgb(0,0,0)'},
        showlegend=False,
        legendgroup=label
    ))


def plotly_baseline(fig: go.Figure, baseline: BaselineCorrection, config: PlotConfig) -> go.Figure:
    if fig is None:
        fig = go.Figure()

    x, y = baseline.x, baseline.y

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name="baseline",
            connectgaps=config.signal_connect_gaps,
        )
    )

    return fig


# def layout_3D(fig: go.Figure):
#     fig.update_layout(scene=dict(
#         yaxis_title='rxn time (sec)',
#         xaxis_title='wavenumber (cm-1)',
#         zaxis_title='signal'),
#     )
#
#
# def plotly_signal_array_3D(array: SignalArray, config: PlotConfig) -> go.Figure:
#     fig = go.Figure()
#
#     for i, t in enumerate(times):
#         fig.add_trace(
#             go.Scatter3d(
#                 x=wavenumber,
#                 y=t * np.ones_like(wavenumber),
#                 z=data[i, :],
#                 mode="lines",
#                 line={"color": "black"},
#                 legendgroup="lines",
#                 showlegend=False if i != 0 else True,
#             )
#         )
#
#     return fig
#
#
# def plotly_signal_array_surface(array: SignalArray, config: PlotConfig) -> go.Figure:
#     fig = go.Figure()
#
#     fig.add_trace(
#         go.Surface(
#             x=wavenumber,
#             y=times,
#             z=data,
#             legendgroup="surface",
#             showlegend=True,
#             showscale=False
#         )
#     )
#
#     return fig
#
# def plotly_signal_array(fig: go.Figure, array: SignalArray, config: PlotConfig):
#     if fig is None:
#         fig = go.Figure()
#
#     signals = config.get_y(array.signals, array.y)
#     for signal in signals:
#         plotly_signal(fig, signal, config)
#     plotly_layout(fig, signal, config)
#     return fig
#
#
# def plotly_chromatogram(chromatogram: Chromatogram, config: PlotConfig):
#     fig = go.Figure()
#     for signal in chromatogram.signals:
#         plotly_signal(fig, signal, config)
#     plotly_layout(fig, signal, config)
#     return fig