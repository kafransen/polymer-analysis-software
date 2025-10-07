
import numpy as np
import plotly.graph_objs as go

import chem_analysis as ca

cal_RI = ca.sec.ConventionalCalibration(lambda time: 10 ** (-0.6 * time + 10.644),
                                        mw_bounds=(160, 1_090_000), name="RI calibration")


def process_one(time_: np.ndarray, y: np.ndarray, output: bool = False):
    sig = ca.sec.SECSignal(x_raw=time_, y_raw=y, calibration=cal_RI)
    sig.processor.add(ca.processing.baselines.ImprovedAsymmetricLeastSquared(
        lambda_=1e6,
        p=0.15,
            weights=[
                ca.processing.weigths.Spans(((6, 9), (16.8, 17.2)), invert=True),
            ]
    ))
    peaks = ca.analysis.peak_picking.max_find_peaks(sig, weights=ca.processing.weigths.Spans((10, 12), invert=True))
    peaks = ca.analysis.boundary_detection.rolling_ball(peaks, n=25, min_height=0.05)

    if output:
        print(peaks.stats_table())

        fig_base = go.Figure()
        fig_base = ca.plot.signal_raw(sig, fig=fig_base)
        fig_base = ca.plot.signal(sig, fig=fig_base)
        fig_base = ca.plot.baseline(sig, fig=fig_base)
        fig_base.layout.yaxis.range = (-1, 50)

        fig = ca.plot.signal(sig)
        fig = ca.plot.peaks(peaks, fig=fig)
        fig.layout.yaxis.range = (-1, 50)
        # fig.write_html("temp.html", auto_open=True)

        ca.plot.plotly_helpers.merge_html_figs([fig_base, fig])

    return peaks.stats_table()


def process_many(time_: np.ndarray, y: np.ndarray):
    table = None
    for i in range(y.shape[1]):
        if table is None:
            table = process_one(time_, y[:, i])
        else:
            table.join(process_one(time_, y[:, i]))

    print(table.to_csv_str())


def main():
    path = r"G:\Other computers\My Laptop\post_doc_2022\Data\polymerizations\DW2-9\DW2-9_SEC.csv"
    csv = np.loadtxt(path, delimiter=",")
    time_ = csv[:, 0]

    process_one(time_, csv[:, -1], output=True)
    # process_many(time_, csv[:, 1:])


if __name__ == "__main__":
    main()
