
import numpy as np

import chem_analysis as ca

cal_RI = ca.sec.ConventionalCalibration(lambda time: 10 ** (-0.6 * time + 10.644),
                                        mw_bounds=(160, 1_090_000), name="RI calibration")


def main():
    path = r"G:\Other computers\My Laptop\post_doc_2022\Data\polymerizations\DW2-8\DW2-8-GPC.csv"
    csv = np.loadtxt(path, delimiter=",")
    time_ = csv[:, 0]
    y = csv[:, 6]
    sig = ca.sec.SECSignal(x_raw=time_, y_raw=y, calibration=cal_RI)
    sig.processor.add(ca.processing.baselines.ImprovedAsymmetricLeastSquared(
        lambda_=1e6,
        p=0.15,
            weights=[
                ca.processing.weigths.Spans(((8, 9.2), (16.8, 17.2)), invert=True),
            ]
    ))
    peaks = ca.analysis.peak_picking.max_find_peaks(sig, weights=ca.processing.weigths.Spans((10, 12), invert=True))
    peaks = ca.analysis.boundary_detection.rolling_ball(peaks, n=25)
    print(peaks.stats_table())

    fig = ca.sec.plot_signal(sig)
    fig = ca.sec.plot_peaks(peaks, fig=fig)
    fig.layout.yaxis.range = (-1, 5)
    fig.write_html("temp.html", auto_open=True)


if __name__ == "__main__":
    main()
