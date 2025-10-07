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
    I_benzene = ca.analysis.integrate.integrate(nmr_array, x_range=(7.1, 7.45))
    I_bond = ca.analysis.integrate.integrate(nmr_array, x_range=(5.6, 6.6))
    I_mono = ca.analysis.integrate.integrate(nmr_array, x_range=(3.4, 3.75))

    mask = I_benzene > 0.01

    conv = np.zeros_like(I_bond)
    conv[mask] = 1 - (I_bond[mask]/I_benzene[mask])/(I_bond[-1]/I_benzene[-1])

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
    path = pathlib.Path(r"C:\Users\nicep\Desktop\post_doc_2022\Data\polymerizations\DW2-14\DW2_14_NMR.feather")
    nmr_array = ca.nmr.NMRSignalArray.from_file(path)

    ## process data
    nmr_array.processor.add(ca.processing.translations.AlignMax(range_=(7.2, 7.4), x_value=7.3))
    nmr_array.processor.add(ca.processing.smoothing.Gaussian(sigma=20))
    nmr_array.delete([123, 99, 72, 67, 1, 0])
    # nmr_array.to_feather(r"C:\Users\nicep\Desktop\post_doc_2022\Data\polymerizations\DW2-14\DW2_14_NMR_proc.feather")

    integrate_array(nmr_array)
    # conv_from_normal(nmr_array)

    ## single analysis
    signal = nmr_array.get_signal(14)
    # signal.to_csv(r"G:\Other computers\My Laptop\post_doc_2022\Data\polymerizations\DW2-9\DW2_9_NMR_14.csv")
    fig = ca.plot.signal(signal)

    fig.write_html("temp.html", auto_open=True)


def make_gif():
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    data = ca.ir.IRSignalArray.from_file(
        r"C:\Users\nicep\Desktop\post_doc_2022\Data\polymerizations\DW2-14\DW2_14_NMR.feather"
    )
    # data.processor.add(ca.processing.re_sampling.CutSpans([2.2,2.7]))
    data.processor.add(ca.processing.smoothing.GaussianTime(sigma=5))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x, y = np.meshgrid(data.x, data.time)

    # Scatter plot
    sc = ax.plot_surface(x, y, data.data, cmap='viridis', vmin=0, vmax=0.2)
    ax.set_xlabel(data.x_label)
    ax.set_ylabel(data.y_label)
    ax.set_zlabel(data.z_label)
    ax.set_zlim(0, 3)

    # Add color bar
    # cbar = fig.colorbar(sc)
    # cbar.set_label('Absorption')

    # Function to rotate the plot
    def update(num, ax, fig):
        ax.view_init(elev=10., azim=num)

    # Generate animation
    rotation_animation = FuncAnimation(fig, update, frames=np.arange(0, 360, 1), fargs=(ax, fig))

    # Save animation as gif
    rotation_animation.save("nmr_data.gif", writer='imagemagick', fps=30, dpi=200, savefig_kwargs={'transparent': True})

    plt.show()


if __name__ == "__main__":
    # main()
    make_gif()
