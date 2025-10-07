
import numpy as np

import chem_analysis as ca


def main2():
    def cal_func(x: np.ndarray) -> np.ndarray:
        return 10**(-0.6035623045394646*x + 10.70478909408625)
    cal = ca.sec.ConventionalCalibration(cal_func, time_bounds=[8.377, 13.1])
    data = ca.sec.SECSignalArray.from_file(
        r"G:\Other computers\My Laptop\post_doc_2022\Data\polymerizations\DW2-7\DW2_7_SEC.feather",
        calibration=cal
    )

    stats = ""
    for i in range(data.time.size): # data.x.size
        signal = data.get_signal(i)
        weights = ca.processing.weights.AdaptiveDistanceMedian(amount=0.5, normalized=False)
        baseline_corr = ca.processing.baselines.ImprovedAsymmetricLeastSquared(lambda_=1e6, weights=weights)
        signal.processor.add(baseline_corr)

        result_peaks_max = ca.analysis.peak_picking.scipy_find_peaks(signal, height=2, distance=500, prominence=1)
        result_peaks = ca.analysis.boundary_detection.rolling_ball(result_peaks_max, n=40, min_height=0.03)

        if len(result_peaks) == 0:
            print(i, "no peaks")
            stats += f"\n{i} No peaks"

        else:
            if stats == "":
                stats += "time,signal," + ",".join(result_peaks.stats_table().headers)
            stats += f"\n{signal.time},{i}," + result_peaks.stats_table().to_csv(with_headers=False)
            print(i, len(result_peaks.peaks))

        fig = ca.sec.plot_calibration(signal.calibration)
        fig = ca.sec.plot_peaks(result_peaks, fig=fig)
        fig = ca.sec.plot_signal(signal, fig=fig)
        fig.write_image(f"plots/{i}.png")

    with open("table.csv", "w", encoding="UTF-8") as f:
        f.write(stats)


def main():
    def cal_func(x: np.ndarray) -> np.ndarray:
        return 10**(-0.6035623045394646*x + 10.70478909408625)
    cal = ca.sec.ConventionalCalibration(cal_func, time_bounds=[8.377, 13.1])
    data = ca.sec.SECSignalArray.from_file(
        r"G:\Other computers\My Laptop\post_doc_2022\Data\polymerizations\DW2-7\DW2_7_SEC.feather",
        calibration=cal
    )

    # single
    signal = data.get_signal(43)

    weights = ca.processing.weights.AdaptiveDistanceMedian(amount=0.5, normalized=False)
    baseline_corr = ca.processing.baselines.ImprovedAsymmetricLeastSquared(lambda_=1e6, weights=weights)
    signal.processor.add(baseline_corr)

    import plotly.graph_objs as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=signal.x_raw, y=signal.y_raw, name="raw"))
    fig.add_trace(go.Scatter(x=signal.x, y=signal.y, name="processed"))
    fig.add_trace(go.Scatter(x=baseline_corr.x, y=baseline_corr.y, name="baseline"))
    fig.write_html("temp.html", auto_open=True)

    result_peaks_max = ca.analysis.peak_picking.scipy_find_peaks(signal, height=2, distance=500, prominence=1)
    if len(result_peaks_max.indexes) == 0:
        print("no peaks")
    result_peaks = ca.analysis.boundary_detection.rolling_ball(result_peaks_max, n=40, min_height=0.03)

    fig = ca.sec.plot_calibration(signal.calibration)
    fig = ca.sec.plot_peaks(result_peaks, fig=fig)
    fig = ca.sec.plot_signal(signal, fig=fig)
    fig.write_html("temp2.html", auto_open=True)


if __name__ == "__main__":
    # main()
    main2()
