
import numpy as np
import plotly.graph_objs as go

from chem_analysis.utils.feather_format import feather_to_numpy, unpack_time_series
from chem_analysis.utils.math import get_slice
from chem_analysis.analysis.line_fitting import DistributionNormalPeak, peak_deconvolution


def conv_from_integration(data, range_1: list, range_2: list):
    slice_1 = get_slice(data.x, range_1[0], range_1[1])
    area_1 = np.trapz(x=data.x[slice_1], y=data.data[:, slice_1])

    slice_2 = get_slice(data.x, range_2[0], range_2[1])
    area_2 = np.trapz(x=data.x[slice_2], y=data.data[:, slice_2])

    return area_2 / (area_1 + area_2)


def conv_from_normal(data):
    range_ = [3.4, 3.9]
    slice_ = get_slice(data.x, start=range_[0], end=range_[1])
    x = data.x[slice_]

    peaks = [
        DistributionNormalPeak(x, 1, 3.53, 0.01),
        DistributionNormalPeak(x, 1, 3.65, 0.01),
    ]
    result = peak_deconvolution(peaks=peaks, xdata=x, ydata=data.data[-1, slice_])
    peaks = result.peaks

    areas = np.empty((data.time.size, 2), dtype=np.float64)
    issues = []
    good = []
    for i, row in enumerate(data.data):
        try:
            y = data.data[i, slice_]
            result = peak_deconvolution(peaks=peaks, xdata=x, ydata=y)
            areas[i] = list(peak.stats.area for peak in result.peaks)
        except Exception as e:
            issues.append(i)
            print(i, e)
            continue

        good.append(i)

    print("issues: ", len(issues), "| total:", len(data.time))
    areas = areas[good]
    times_ = data.time_zeroed[good]

    plot_fit(data.x, data.data[-1], result.peaks)

    return np.column_stack((times_, areas[:, 0] / (areas[:, 0] + areas[:, 1])))


def plot_fit(x, y, peaks):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y))
    for peak in peaks:
        fig.add_trace(go.Scatter(x=peak.x, y=peak.y))

    # fig.show()
    fig.write_html("spectra.html", auto_open=True)


class NMRData:
    def __init__(self, ppm, times, data):
        self.time = times
        self.time_zeroed = self.time - self.time[0]
        self.pmm = ppm
        self.data = data

    @property
    def x(self):
        return self.pmm


def conv_analysis(data: NMRData):
    conv_integration = conv_from_integration(data, range_1=[3.65, 3.8], range_2=[3.488, 3.65])
    print("integration:", conv_integration[-1])
    conv_normal = conv_from_normal(data)
    print("normal:", conv_normal[-1, 1])
    # conv_cauchy = conv_from_fit(data, range_=range_, func=distribution_cauchy)
    # conv_voigt = conv_from_fit(data, range_=range_, func=distribution_volt)


    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.time_zeroed, y=conv_integration, name="conv_int"))
    fig.add_trace(go.Scatter(x=conv_normal[:, 0], y=conv_normal[:, 1], name="conv_normal"))
    # fig.add_trace(go.Scatter(x=data.time_zeroed, y=conv_cauchy, name="conv_cauchy"))
    # fig.add_trace(go.Scatter(x=data.time_zeroed, y=conv_voigt, name="conv_voigt"))
    fig.add_trace(go.Scatter(x=[data.time_zeroed[0], data.time_zeroed[-1]], y=[conv_integration[-1], conv_integration[-1]], name="true"))
    fig.add_trace(
        go.Scatter(x=[data.time_zeroed[0], data.time_zeroed[-1]], y=[conv_normal[-1, 1], conv_normal[-1, 1]],
                   name="true_normal"))
    fig.update_layout(autosize=False, width=800, height=600, font=dict(family="Arial", size=18, color="black"),
                      plot_bgcolor="white", showlegend=True)
    fig.update_xaxes(title="<b>rxn time (min)</b>", tickprefix="<b>", ticksuffix="</b>", showline=True,
                     linewidth=5, mirror=True, linecolor='black', ticks="outside", tickwidth=4, showgrid=False,
                     gridwidth=1, gridcolor="lightgray")
    fig.update_yaxes(title="<b>conversion</b>", tickprefix="<b>", ticksuffix="</b>", showline=True,
                     linewidth=5, mirror=True, linecolor='black', ticks="outside", tickwidth=4, showgrid=False,
                     gridwidth=1, gridcolor="lightgray", range=[0, 1])
    # fig.show()
    fig.write_html("temp.html", auto_open=True)


def plot_spectra(data):
    n = [0, 2, 5, 10, 20, 30, 200, -1]
    fig = go.Figure()
    for i in n:
        fig.add_trace(go.Scatter(x=data.x, y=data.data[i]))

    # fig.show()
    fig.write_html("spectra.html", auto_open=True)


def clean_data(data):
    remove_spectra = []

    # remove spectra with no signals
    for i in range(len(data.time)):
        if np.max(data.data[i]) < 1:
            remove_spectra.append(i)

    data.time = np.delete(data.time, remove_spectra)
    data.time_zeroed = np.delete(data.time_zeroed, remove_spectra)
    data.data = np.delete(data.data, remove_spectra, axis=0)


def main():
    path = r"G:\Other computers\My Laptop\post_doc_2022\Data\polymerizations\DW_flow\DW2_flow_rate_NMR.feather"
    #r"C:\Users\nicep\Desktop\DW2_flow_rate_NMR.feather"
    data_ = feather_to_numpy(path)
    # data = np.vstack((data_[0, :], data_[30:, :]))
    data = NMRData(*unpack_time_series(data_))
    clean_data(data)

    # plot_spectra(data)
    conv_analysis(data)


if __name__ == "__main__":
    main()
