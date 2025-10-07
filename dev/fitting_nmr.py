

import numpy as np
from scipy.optimize import curve_fit
import plotly.graph_objs as go

from chem_analysis.utils.math import get_slice


def normal_distribution(x, amp=1, mean=1, sigma=1):
    return amp / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mean) ** 2 / (2 * sigma ** 2))


def bimodal(x, amp1, mean1, sigma1, amp2, mean2, sigma2):
    return normal_distribution(x, amp1, mean1, sigma1) + normal_distribution(x, amp2, mean2, sigma2)


def do_fitting(x, y, range_, peaks):
    # initial guesses
    slice_ = get_slice(x, start=range_[0], end=range_[1])
    amp1 = np.max(y[slice_])
    mean1 = peaks[0]
    sigma1 = 0.01
    amp2 = np.max(y[slice_])
    mean2 = peaks[1]
    sigma2 = 0.01

    params, covariance = curve_fit(bimodal, x, y, p0=[amp1, mean1, sigma1, amp2, mean2, sigma2])
    amp1, mean1, sigma1, amp2, mean2, sigma2 = params

    return amp1, mean1, sigma1, amp2, mean2, sigma2


def gaussian_area(x, amp, mean, sigma):
    return np.trapz(x=x, y=normal_distribution(x, amp, mean, sigma))
    # return 2.5066282746310002*amp*abs(sigma)  # math.sqrt(2* math.pi)


def fitting(data):
    range_ = [3.4, 3.9]
    peaks = [3.53, 3.65]

    params = np.empty((data.time.size, 6), dtype=np.float64)
    areas = np.empty((data.time.size, 2), dtype=np.float64)
    issues = []
    good = []
    for i, row in enumerate(data.data):
        try:
            params[i, :] = do_fitting(data.x, data.data[i, :], range_, peaks)
        except:
            issues.append(i)
            continue
        areas[i, 0] = gaussian_area(data.x, *params[i, :3])
        areas[i, 1] = gaussian_area(data.x, *params[i, 3:])
        good.append(i)

    print(len(issues), len(data.time))
    areas = areas[good]
    conv = areas[:, 1] / (areas[:, 0] + areas[:, 1])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.time_zeroed[good], y=conv))
    fig.update_layout(autosize=False, width=800, height=600, font=dict(family="Arial", size=18, color="black"),
                      plot_bgcolor="white", showlegend=False)
    fig.update_xaxes(title="<b>rxn time (min)</b>", tickprefix="<b>", ticksuffix="</b>", showline=True,
                     linewidth=5, mirror=True, linecolor='black', ticks="outside", tickwidth=4, showgrid=False,
                     gridwidth=1, gridcolor="lightgray")
    fig.update_yaxes(title="<b>conversion</b>", tickprefix="<b>", ticksuffix="</b>", showline=True,
                     linewidth=5, mirror=True, linecolor='black', ticks="outside", tickwidth=4, showgrid=False,
                     gridwidth=1, gridcolor="lightgray", range=[0, 1])
    fig.show()

    return np.column_stack((data.time, conv))





def fit_check(data):
    range_ = [3.4, 3.9]
    peaks = [3.54, 3.64]

    i=100
    params = do_fitting(data.x, data.data[i, :], range_, peaks)
    print(params)
    area1 = gaussian_area(data.x, *params[:3])
    area2 = gaussian_area(data.x, *params[3:])
    print("conv:", area2 / (area1 + area2))


    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.x, y=data.data[i, :], name="data"))
    fig.add_trace(go.Scatter(x=data.x, y=normal_distribution(data.x, *params[:3]), name="fit_1"))
    fig.add_trace(go.Scatter(x=data.x, y=normal_distribution(data.x, *params[3:]), name="fit_2"))
    fig.add_trace(go.Scatter(x=data.x, y=normal_distribution(data.x, *params[:3])+normal_distribution(data.x, *params[3:]), name="fit_full"))
    fig.update_layout(autosize=False, width=800, height=600, font=dict(family="Arial", size=18, color="black"),
                      plot_bgcolor="white", showlegend=True)
    fig.update_xaxes(title="<b>rxn time (min)</b>", tickprefix="<b>", ticksuffix="</b>", showline=True,
                     linewidth=5, mirror=True, linecolor='black', ticks="outside", tickwidth=4, showgrid=False,
                     gridwidth=1, gridcolor="lightgray", range=range_)
    fig.update_yaxes(title="<b>conversion</b>", tickprefix="<b>", ticksuffix="</b>", showline=True,
                     linewidth=5, mirror=True, linecolor='black', ticks="outside", tickwidth=4, showgrid=False,
                     gridwidth=1, gridcolor="lightgray")
    fig.show()


def main():
    fit_check(nmr_data)
    nmr_conv2 = fitting(nmr_data)


if __name__ == "__main__":
    main()
