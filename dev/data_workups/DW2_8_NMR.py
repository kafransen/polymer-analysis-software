import pathlib

import numpy as np
import plotly.graph_objs as go

import chem_analysis as ca
from chem_analysis.utils.math import get_slice
from chem_analysis.analysis.line_fitting import DistributionNormalPeak, peak_deconvolution


def conv_from_normal(nmr_array):
    range_ = [3.4, 3.9]
    slice_ = get_slice(nmr_array.x, start=range_[0], end=range_[1])
    x = nmr_array.x[slice_]

    peaks = [
        DistributionNormalPeak(x, 1, 3.65, 0.01),
        DistributionNormalPeak(x, 1, 3.76, 0.01),
    ]
    # result = peak_deconvolution(peaks=peaks, xdata=x, ydata=nmr_array.data[-1, slice_])
    # peaks = result.peaks

    areas = np.empty((nmr_array.time.size, 2), dtype=np.float64)
    issues = []
    good = []
    for i, row in enumerate(nmr_array.data):
        try:
            y = nmr_array.data[i, slice_]
            result = peak_deconvolution(peaks=peaks, xdata=x, ydata=y)
            areas[i] = list(peak.stats.area for peak in result.peaks)
        except Exception as e:
            issues.append(i)
            print(i, e)
            continue

        good.append(i)

    print("issues: ", len(issues), "| total:", len(nmr_array.time))
    areas = areas[good]
    times_ = nmr_array.time_zeroed[good]

    plot_fit(nmr_array.x, nmr_array.data[-1], result.peaks)

    conv = areas[:, 0] / (areas[:, 0] + areas[:, 1])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times_, y=conv,
                             text=[f'X: {x}, Index: {i}' for i, x in enumerate(times_)],
                             hoverinfo='text',
                             ))
    fig.write_html("conv.html", auto_open=True)


def plot_fit(x, y, peaks):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y))
    for peak in peaks:
        fig.add_trace(go.Scatter(x=peak.x, y=peak.y))

    # fig.show()
    fig.write_html("spectra.html", auto_open=True)


def integrate_array(nmr_array: ca.nmr.NMRSignalArray):
    I_bond = ca.analysis.integrate.integrate(nmr_array, x_range=(6.11, 6.4))
    mask = I_bond < 0.05
    I_benzene = ca.analysis.integrate.integrate(nmr_array, x_range=(7.2, 7.5))
    I_ratio = I_bond / I_benzene
    I_ratio[mask] = np.max(I_ratio)

    I_poly = ca.analysis.integrate.integrate(nmr_array, x_range=(3.52, 3.72))
    mask = I_poly < 0.05
    I_poly[mask] = 0
    I_mon = ca.analysis.integrate.integrate(nmr_array, x_range=(3.72, 3.83))
    # mask = I_mon < 0.05
    # I_mon[mask] = 0

    # print
    conv = I_poly / (I_mon + I_poly)
    for i in range(len(nmr_array.time_zeroed)):
        print(nmr_array.time_zeroed[i], conv[i])

    mask = np.logical_not(conv == 0)
    fig = go.Figure(go.Scatter(x=nmr_array.time_zeroed, y=1 - I_ratio / np.max(I_ratio)))
    fig.add_trace(go.Scatter(x=nmr_array.time_zeroed[mask], y=conv[mask],
                             text=[f'X: {x}, Index: {i}' for i, x in enumerate(nmr_array.time_zeroed)],
                             hoverinfo='text',
                             ))
    fig.write_html("conv.html", auto_open=True)


def main():
    path = pathlib.Path(r"G:\Other computers\My Laptop\post_doc_2022\Data\polymerizations\DW2-8\DW2_8_NMR2.feather")
    nmr_array = ca.nmr.NMRSignalArray.from_file(path)
    nmr_array.processor.add(ca.processing.translations.AlignMax(range_=(2.2, 2.7)))
    nmr_array.processor.add(ca.processing.smoothing.Gaussian(sigma=15))
    # nmr_array.to_feather(r"G:\Other computers\My "
    #                      r"Laptop\post_doc_2022\Data\polymerizations\DW2-8\DW2_8_NMR2_proc.feather")
    integrate_array(nmr_array)
    conv_from_normal(nmr_array)

    signal = nmr_array.get_signal(5)
    fig = ca.plot.signal(signal)
    fig.write_html("temp.html", auto_open=True)


if __name__ == "__main__":
    main()
