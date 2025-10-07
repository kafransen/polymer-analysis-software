
import numpy as np
import plotly.graph_objs as go

import chem_analysis as ca

cal_RI = ca.sec.ConventionalCalibration(lambda time: 10 ** (-0.6 * time + 10.644),
                                        mw_bounds=(160, 1_090_000), name="RI calibration")


def process_one(sig: ca.sec.SECSignal, *, output: bool = False, save_img: bool = False):
    # sig = ca.sec.SECSignal(x_raw=time_, y_raw=y, calibration=cal_RI)
    sig.processor.add(ca.processing.baselines.ImprovedAsymmetricLeastSquared(
        lambda_=1e6,
        p=0.15,
            weights=[
                ca.processing.weigths.Spans(((6, 9), (16.8, 17.2)), invert=True),
            ]
    ))
    peaks = ca.analysis.peak_picking.max_find_peaks(sig, weights=ca.processing.weigths.Spans((10, 12.2), invert=True))
    peaks = ca.analysis.boundary_detection.rolling_ball(peaks, n=45, min_height=0.05, n_points_with_pos_slope=1)

    if output:
        print(peaks.stats_table())

        fig_base = go.Figure()
        fig_base = ca.plot.signal_raw(sig, fig=fig_base)
        fig_base = ca.plot.signal(sig, fig=fig_base)
        fig_base = ca.plot.baseline(sig, fig=fig_base)
        fig_base.layout.yaxis.range = (-1, 50)

        fig = ca.plot.signal(sig)
        fig = ca.plot.peaks(peaks, fig=fig)
        fig = ca.plot.calibration(sig.calibration, fig=fig)
        fig.layout.yaxis.range = (-1, 50)

        ca.plot.plotly_helpers.merge_html_figs([fig_base, fig])

    if save_img:
        fig = ca.plot.signal(sig)
        fig = ca.plot.peaks(peaks, fig=fig)
        fig = ca.plot.calibration(sig.calibration, fig=fig)
        fig.layout.yaxis.range = (-1, 50)
        fig.write_image(f"img/signal{sig.id_}.png")

    return peaks.stats_table()


def process_many(signals: list[ca.sec.SECSignal], *, output: bool = False, save_img: bool = False):
    table = None
    for sig in signals:
        if table is None:
            table = process_one(sig, output=output, save_img=save_img)
        else:
            table.join(process_one(sig, output=output, save_img=save_img))
        print(f"{sig.id_} done")

    print(table.to_csv_str())


def create_gif(data: ca.base_obj.SignalArray):
    from plotly_gif import GIF, capture

    gif = GIF()

    for i in range(len(data.time)):
        sig = data.get_signal(i, processed=True)
        fig = ca.plot.signal(sig)
        fig.layout.xaxis.title = "<b>retention time (min)</b>"
        fig.layout.yaxis.title = "<b>signal</b>"
        fig.layout.yaxis.range = (-1, 50)
        fig.layout.xaxis.range = (8, 15)
        gif.create_image(fig)  # create_gif image for gif

    gif.create_gif(length=30000)  # generate gif


def main():
    path = r"C:\Users\nicep\Desktop\post_doc_2022\Data\polymerizations\DW2-14\DW2-14_RI.npy"
    array = ca.sec.SECSignalArray.from_file(path, calibration=cal_RI)
    # create_gif(array)

    # process_one(array.get_signal(10), output=True)
    process_many([array.get_signal(i) for i in range(len(array.time))], save_img=True)


if __name__ == "__main__":
    main()
