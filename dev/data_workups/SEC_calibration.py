
import numpy as np
import plotly.graph_objs as go

import chem_analysis as ca

yellow = [
    2_110_000,
    427_000,
    37_900,
    5_970,
    589
]

green = [
    1_090_000,
    190_000,
    18_100,
    2_420
]


def main2():
    yellow_cal = [
        [427_000, 8.377],
        [37_900, 10.19],
        [5_970, 11.51],
        [589, 13.1],
    ]

    green_cal = [
        [190_000, 8.948],
        [18_100, 10.76],
        [2_420, 12.1],
    ]
    cal_points = np.array(yellow_cal + green_cal)
    sorted_indices = np.argsort(cal_points[:, 0])
    cal_points = cal_points[sorted_indices]
    cal_points[:, [0, 1]] = cal_points[:, [1, 0]]

    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=cal_points[:, 0], y=cal_points[:, 1]))
    # fig.write_html("temp.html", auto_open=True)

    cal_points[:, 1] = np.log10(cal_points[:, 1])

    params = np.polyfit(x=cal_points[:, 0], y=cal_points[:, 1], deg=1)
    func_baseline = np.poly1d(params)
    print(params)
    print(f"y = 10**({params[0]}*x + {params[1]})")
    y_baseline = func_baseline(cal_points[:, 0])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cal_points[:, 0], y=cal_points[:, 1]))
    fig.add_trace(go.Scatter(x=cal_points[:, 0], y=y_baseline))
    fig.write_html("temp2.html", auto_open=True)

    #[ 0.59737886 -2.15986422]


def main():
    data_yellow_RI = ca.sec.SECSignal.from_file(
        r"G:\Other computers\My Laptop\post_doc_2022\Data\polymerizations\calibration\PStQuick_C_RI.csv")
    # data_yellow_RI.processor.add(ca.processing.baseline_correction.Polynomial())

    result_peaks = ca.analysis.peak_picking.scipy_find_peaks(data_yellow_RI, height=20, distance=300)
    result_peaks = ca.analysis.boundry_detection.rolling_ball(result_peaks)
    print(result_peaks.stats_table(sig_figs=4))

    data_green_RI = ca.sec.SECSignal.from_file(
        r"G:\Other computers\My Laptop\post_doc_2022\Data\polymerizations\calibration\PStQuick_D_RI.csv")
    result_peaks2 = ca.analysis.peak_picking.scipy_find_peaks(data_green_RI, height=20, distance=300)
    result_peaks2 = ca.analysis.boundry_detection.rolling_ball(result_peaks2)
    print(result_peaks2.stats_table(sig_figs=4))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data_yellow_RI.x, y=data_yellow_RI.y))
    fig.add_trace(go.Scatter(x=data_green_RI.x, y=data_green_RI.y))
    fig.write_html("temp.html", auto_open=True)


if __name__ == "__main__":
    # main()
    main2()
