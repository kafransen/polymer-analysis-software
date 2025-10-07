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
    I_bond = ca.analysis.integrate.integrate(nmr_array, x_range=(5.6, 6.4))
    I_benzene = ca.analysis.integrate.integrate(nmr_array, x_range=(7.1, 7.5))
    I_poly = ca.analysis.integrate.integrate(nmr_array, x_range=(3, 3.8)) # 3.46
    I_DMSO = ca.analysis.integrate.integrate(nmr_array, x_range=(2.05, 2.8))
    I_poly = I_poly - I_DMSO * 0.01
    slice_ = ca.utils.math.get_slice(nmr_array.x, 7.1, 7.5)
    mask = np.max(nmr_array.data[:, slice_], axis=1) > 0.3

    area_per_h = np.zeros_like(I_bond)
    area_per_h[mask] = I_poly[mask]/I_benzene[mask]
    mean_ = np.mean(area_per_h[mask])
    std_ = np.std(area_per_h[mask])
    mask2 = np.logical_and(area_per_h < mean_+std_, area_per_h > mean_-std_)
    mask = np.logical_and(mask, mask2)

    conv = np.zeros_like(I_bond)
    conv[mask] = 1 - I_bond[mask]/I_benzene[mask]/area_per_h[mask]

    # print
    c = conv[mask]
    t = nmr_array.time[mask]
    for i in range(len(t)):
        print(t[i], c[i])

    mask = np.logical_not(conv == 0)
    fig = go.Figure(go.Scatter(x=nmr_array.time_zeroed, y=conv,
                             text=[f'X: {x}, Index: {i}' for i, x in enumerate(nmr_array.time_zeroed)],
                             hoverinfo='text',
                             ))
    fig.write_html("conv.html", auto_open=True)


def create_gif(data: ca.base_obj.SignalArray):
    from plotly_gif import GIF, capture

    gif = GIF()

    for i in range(len(data.time)):
        sig = data.get_signal(i, processed=True)
        fig = ca.plot.signal(sig)
        fig.layout.xaxis.title = "<b>ppm(cm-1)</b>"
        fig.layout.yaxis.title = "<b>signal</b>"
        gif.create_image(fig)  # create_gif image for gif

    gif.create_gif(length=30000)  # generate gif


def main():
    path = pathlib.Path(r"C:\Users\nicep\Desktop\post_doc_2022\Data\polymerizations\DW2-10\DW2_10_NMR.feather")
    nmr_array = ca.nmr.NMRSignalArray.from_file(path)
    nmr_array.processor.add(ca.processing.translations.AlignMax(range_=(2.2, 2.7)))
    create_gif(nmr_array)
    nmr_array.processor.add(ca.processing.smoothing.Gaussian(sigma=35))
    # nmr_array.to_feather(r"G:\Other computers\My "
    #                      r"Laptop\post_doc_2022\Data\polymerizations\DW2-8\DW2_8_NMR2_proc.feather")

    # ca.plotting.array_dynamic(nmr_array)

    integrate_array(nmr_array)
    # conv_from_normal(nmr_array)

    signal = nmr_array.get_signal(14)
    # signal.to_csv(r"G:\Other computers\My Laptop\post_doc_2022\Data\polymerizations\DW2-9\DW2_9_NMR_14.csv")
    fig = ca.plot.signal(signal)
    fig.write_html("temp.html", auto_open=True)


if __name__ == "__main__":
    main()
