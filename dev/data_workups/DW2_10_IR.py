import pathlib

import plotly.graph_objs as go
import numpy as np
import chem_analysis as ca
from chem_analysis.utils.math import normalize_by_max
from chem_analysis.plotting.plotly_helpers import merge_html_figs


def conversion(mca_result) -> np.ndarray:
    return mca_result.C[:, 1] / (mca_result.C[:, 0] + mca_result.C[:, 1])


def plot_mca_results(mcrar, x, D, times, conv):
    # resulting spectra
    fig1 = go.Figure()
    for i in range(mcrar.ST.shape[0]):
        fig1.add_trace(go.Scatter(y=normalize_by_max(mcrar.ST[i, :]), x=x, name=f"mca_{i}"))
    fig1.add_trace(go.Scatter(y=normalize_by_max(D[0, :]), x=x, name="early"))
    fig1.add_trace(go.Scatter(y=normalize_by_max(D[int(D.shape[0] / 2), :]), x=x, name="middle"))
    fig1.add_trace(go.Scatter(y=normalize_by_max(D[-1, :]), x=x, name="late"))
    fig1.update_layout(autosize=False, width=1200, height=600, font=dict(family="Arial", size=18, color="black"),
                       plot_bgcolor="white", showlegend=True)
    fig1.update_xaxes(title="<b>wavenumber (cm-1) (min)</b>", tickprefix="<b>", ticksuffix="</b>", showline=True,
                      linewidth=5, mirror=True, linecolor='black', ticks="outside", tickwidth=4, showgrid=False,
                      gridwidth=1, gridcolor="lightgray", autorange="reversed")
    fig1.update_yaxes(title="<b>normalized absorbance</b>", tickprefix="<b>", ticksuffix="</b>", showline=True,
                      linewidth=5, mirror=True, linecolor='black', ticks="outside", tickwidth=4, showgrid=False,
                      gridwidth=1, gridcolor="lightgray")

    fig2 = go.Figure()
    for i in range(mcrar.C.shape[1]):
        fig2.add_trace(go.Scatter(x=times, y=mcrar.C[:, i], name=f"mca_{i}"))
    fig2.update_layout(autosize=False, width=800, height=600, font=dict(family="Arial", size=18, color="black"),
                       plot_bgcolor="white", showlegend=True)
    fig2.update_xaxes(title="<b>rxn time (min)</b>", tickprefix="<b>", ticksuffix="</b>", showline=True,
                      linewidth=5, mirror=True, linecolor='black', ticks="outside", tickwidth=4, showgrid=False,
                      gridwidth=1, gridcolor="lightgray")
    fig2.update_yaxes(title="<b>conversion</b>", tickprefix="<b>", ticksuffix="</b>", showline=True,
                      linewidth=5, mirror=True, linecolor='black', ticks="outside", tickwidth=4, showgrid=False,
                      gridwidth=1, gridcolor="lightgray", range=[0, 1])

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=times, y=conv))
    fig3.update_layout(autosize=False, width=800, height=600, font=dict(family="Arial", size=18, color="black"),
                       plot_bgcolor="white", showlegend=False)
    fig3.update_xaxes(title="<b>rxn time (min)</b>", tickprefix="<b>", ticksuffix="</b>", showline=True,
                      linewidth=5, mirror=True, linecolor='black', ticks="outside", tickwidth=4, showgrid=False,
                      gridwidth=1, gridcolor="lightgray")
    fig3.update_yaxes(title="<b>conversion</b>", tickprefix="<b>", ticksuffix="</b>", showline=True,
                      linewidth=5, mirror=True, linecolor='black', ticks="outside", tickwidth=4, showgrid=False,
                      gridwidth=1, gridcolor="lightgray", range=[0, 1])

    merge_html_figs([fig1, fig2, fig3], auto_open=True)


def load_pure():
    path = r"G:\Other computers\My Laptop\post_doc_2022\Data\polymerizations"
    path = pathlib.Path(path)
    MA = ca.ir.IRSignal.from_file(path / "ATIR_MA.csv")
    PMA = ca.ir.IRSignal.from_file(path / "ATIR_PMA.csv")
    DMSO = ca.ir.IRSignal.from_file(path / "ATIR_DMSO.csv")
    FL = ca.ir.IRSignal.from_file(path / "ATIR_perflourohexane.csv")

    return MA, PMA, DMSO, FL


def mca_2(data: ca.base.SignalArray):
    t_slice = slice(60, -3)  # ca.utils.general_math.get_slice(data.time_zeroed, start=800)  # slice(0, -1)
    x_slice = ca.utils.general_math.get_slice(data.x, start=760, end=1900)

    mca_times = data.time[t_slice]
    mca_x = data.x[x_slice]
    mca_data = data.data[t_slice, x_slice]

    mask = ca.processing.weigths.Slices(
        [
            ca.utils.general_math.get_slice(mca_x, start=875, end=1100),
            ca.utils.general_math.get_slice(mca_x, start=1350, end=1600),
        ],
    )
    mask = mask.get_mask(mca_x, mca_data[0, :])
    mca_x = mca_x[mask]
    mca_data = mca_data[:, mask]

    D = mca_data
    C = np.ones((D.shape[0], 2)) * .5
    C[0, :] = np.array([.7, 0.3])

    print("working")
    mca = ca.analysis.mca.MultiComponentAnalysis(
        max_iters=200,
        c_constraints=[ca.analysis.mca.ConstraintNonneg(), ca.analysis.mca.ConstraintConv()],
        st_constraints=[],
        tolerance_increase=100
    )
    mca_result = mca.fit(D, C=C, verbose=True)

    conv = conversion(mca_result)
    plot_mca_results(mca_result, mca_x, D, mca_times, conv)
    return np.column_stack((mca_times, conv))


def mca_4(data: ca.base.SignalArray):
    t_slice = slice(60, -3)  # ca.utils.general_math.get_slice(data.time_zeroed, start=800)  # slice(0, -1)
    x_slice = ca.utils.general_math.get_slice(data.x, start=760, end=1900)

    mca_times = data.time[t_slice]
    mca_x = data.x[x_slice]
    mca_data = data.data[t_slice, x_slice]

    # mask = ca.processing.weigths.Slices(
    #     [
    #         ca.utils.general_math.get_slice(mca_x, start=875, end=1100),
    #         ca.utils.general_math.get_slice(mca_x, start=1350, end=1600),
    #     ],
    # )
    # mask = mask.get_mask(mca_x, mca_data[0, :])
    # mca_x = mca_x[mask]
    # mca_data = mca_data[:, mask]

    D = mca_data
    C = mca_4_ST(data)
    C = C[t_slice]

    print("working")
    mca = ca.analysis.mca.MultiComponentAnalysis(
        max_iters=200,
        c_constraints=[
            ca.analysis.mca.ConstraintConv(index=[0, 1]),
            ca.analysis.mca.ConstraintRange(
                [
                    (0, 1),
                    (0, 1),
                    (-0.2, 1),
                    (0, 3),
                ]
            )
        ],
        st_constraints=[],
        tolerance_increase=100
    )
    mca_result = mca.fit(D, C=C, verbose=True)

    conv = conversion(mca_result)
    plot_mca_results(mca_result, mca_x, D, mca_times, conv)
    return np.column_stack((mca_times, conv))


def mca_2_ST(data: ca.base.SignalArray):
    t_slice = slice(80, -4)  # ca.utils.general_math.get_slice(data.time_zeroed, start=800)  # slice(0, -1)
    x_slice = ca.utils.general_math.get_slice(data.x, start=760, end=1900)

    mca_times = data.time[t_slice]
    mca_x = data.x[x_slice]
    mca_data = data.data[t_slice, x_slice]

    mask = ca.processing.weigths.Slices(
        [
            ca.utils.general_math.get_slice(mca_x, start=875, end=1350),
            ca.utils.general_math.get_slice(mca_x, start=1350, end=1600),
        ],
    )
    mask = mask.get_mask(mca_x, mca_data[0, :])
    mca_x = mca_x[mask]
    mca_data = mca_data[:, mask]

    MA, PMA, DMSO, FL = load_pure()

    D = mca_data

    ST = np.ones((2, D.shape[1]))
    ST[0, :] = MA.y[x_slice][mask]
    ST[1, :] = PMA.y[x_slice][mask]

    print("working")
    mca = ca.analysis.mca.MultiComponentAnalysis(
        max_iters=200,
        c_constraints=[ca.analysis.mca.ConstraintNonneg(), ca.analysis.mca.ConstraintConv()],
        st_constraints=[],
        tolerance_increase=100
    )
    mca_result = mca.fit(D, ST=ST, st_fix=[0,1], verbose=True)

    conv = conversion(mca_result)
    plot_mca_results(mca_result, mca_x, D, mca_times, conv)
    return np.column_stack((mca_times, conv))


def mca_5_ST(data: ca.base.SignalArray):
    t_slice = slice(60, -3)  # ca.utils.general_math.get_slice(data.time_zeroed, start=800)  # slice(0, -1)
    x_slice = ca.utils.general_math.get_slice(data.x, start=760, end=1900)

    mca_times = data.time[t_slice]
    mca_x = data.x[x_slice]
    mca_data = data.data[t_slice, x_slice]

    D = mca_data

    MA, PMA, DMSO, FL = load_pure()
    ST = np.ones((5, D.shape[1]))
    ST[0, :] = MA.y[x_slice]
    ST[1, :] = PMA.y[x_slice]
    ST[2, :] = DMSO.y[x_slice] * 0.2
    ST[3, :] = FL.y[x_slice]
    noise = mca_data[-1]
    noise = noise / np.max(noise)
    mask = noise > 0.2
    noise[mask] = .2
    mask = noise < -0.2
    noise[mask] = -0.2
    ST[4, :] = noise

    print("working")
    mca = ca.analysis.mca.MultiComponentAnalysis(
        max_iters=200,
        c_constraints=[
            ca.analysis.mca.ConstraintConv(index=[0, 1]),
            ca.analysis.mca.ConstraintRange(
                [
                    (0, 1),
                    (0, 1),
                    (-3, 0),
                    (0, 3),
                    (0.01, 0.3)
                ]
            )
        ],
        st_constraints=[ca.analysis.mca.ConstraintNonneg(index=[4])],
        tolerance_increase=100
    )
    mca_result = mca.fit(D, ST=ST, st_fix=[0, 1, 2, 3, 4], verbose=True)

    conv = conversion(mca_result)
    plot_mca_results(mca_result, mca_x, D, mca_times, conv)
    return np.column_stack((mca_times, conv))


def mca_4_ST(data: ca.base.SignalArray):
    t_slice = slice(80, -4)  # ca.utils.general_math.get_slice(data.time_zeroed, start=800)
    x_slice = ca.utils.general_math.get_slice(data.x, start=760, end=1900)

    mca_times = data.time[t_slice]
    mca_x = data.x[x_slice]
    mca_data = data.data[t_slice, x_slice]

    # mask = ca.processing.weigths.Slices(
    #     [
    #         ca.utils.general_math.get_slice(mca_x, start=875, end=1100),
    #         ca.utils.general_math.get_slice(mca_x, start=1350, end=1500),
    #     ],
    # )
    # mask = mask.get_mask(mca_x, mca_data[0, :])
    # mca_x = mca_x[mask]
    # mca_data = mca_data[:, mask]

    D = mca_data
    MA, PMA, DMSO, FL = load_pure()
    ST = np.ones((4, D.shape[1]))
    ST[0, :] = MA.y[x_slice]
    ST[1, :] = PMA.y[x_slice]
    ST[2, :] = DMSO.y[x_slice]
    ST[3, :] = FL.y[x_slice]

    print("working")
    mca = ca.analysis.mca.MultiComponentAnalysis(
        max_iters=200,
        c_constraints=[
            ca.analysis.mca.ConstraintConv(index=[0, 1]),
            ca.analysis.mca.ConstraintRange(
                [
                    (0, 1),
                    (0, 1),
                    (-0.2, 1),
                    (0, 3),
                ]
            )
        ],
        st_constraints=[],
        tolerance_increase=100
    )
    mca_result = mca.fit(D, ST=ST, st_fix=[0, 1, 2, 3], verbose=True)

    conv = conversion(mca_result)
    plot_mca_results(mca_result, mca_x, D, mca_times, conv)
    return np.column_stack((mca_times, conv))


def create_gif(data: ca.base_obj.SignalArray):
    from plotly_gif import GIF, capture

    gif = GIF()

    for i in range(150, len(data.time)-10):
        sig = data.get_signal(i, processed=True)
        fig = ca.plot.signal(sig)
        fig.layout.xaxis.title = "<b>wavenumber(cm-1)</b>"
        fig.layout.yaxis.title = "<b>absorbtion</b>"
        gif.create_image(fig)  # create_gif image for gif

    gif.create_gif()  # generate gif


def main():
    data = ca.ir.IRSignalArray.from_file(
        r"C:\Users\nicep\Desktop\post_doc_2022\Data\polymerizations\DW2-10\DW2_10_IR.feather"
    )
    data.processor.add(ca.processing.baselines.SubtractOptimize(data.data[-2, :]))
    data.pop(-2)
    # data.pop(-1)
    # data.to_feather(r"G:\Other computers\My Laptop\post_doc_2022\Data\polymerizations\DW2-10\DW2_10_IR_fix.feather")

    # signal = data.get_signal(300)
    # fig = ca.plot.signal(signal)
    # fig.add_trace(go.Scatter(x=signal.x_raw, y=signal.y_raw))
    # fig.write_html("temp.html", auto_open=True)

    data.processor.add(ca.processing.re_sampling.CutOffValue(x_span=529, cut_off_value=0.015))
    # data.processor.add(ca.processing.translations.ScaleMax(range_=(1700, 1800)))
    # data.processor.add(ca.processing.smoothing.GaussianTime(sigma=3))
    # data.processor.add(ca.processing.smoothing.ExponentialTime(a=0.7))
    # data.processor.add(ca.processing.baselines.Polynomial(
    #         degree=1,
    #         weights=ca.processing.weigths.AdaptiveDistanceMedian(threshold=0.2)
    # ))
    # data.processor.add(ca.processing.smoothing.Gaussian(sigma=2))
    # data.processor.add(ca.processing.translations.ScaleMax(range_=(1700, 1800)))

    # create_gif(data)

    # mca_result_1 = mca_2_ST(data)
    # for i in range(mca_result_1.shape[0]):
    #     print(mca_result_1[i, 0], mca_result_1[i, 1])


    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=data.x,
            y=data.time_zeroed[:-50],
            z=data.data[:-50],
            legendgroup="surface",
            showlegend=True,
            showscale=False
        )
    )
    from plotly_gif import three_d_scatter_rotate, GIF
    gif = GIF()
    three_d_scatter_rotate(gif, fig, auto_create=False)
    gif.create_gif(length=10_000)
    fig.show()

if __name__ == "__main__":
    main()
