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


def mca_2(data: ca.base.SignalArray):
    t_slice = slice(0, -1)
    x_slice = ca.utils.general_math.get_slice(data.x, start=1500, end=2000)

    mca_times = data.time_zeroed[t_slice]
    mca_x = data.x[x_slice]
    mca_data = data.data[t_slice, x_slice]

    # baseline = ca.processing.baselines.AsymmetricLeastSquared()
    # baseline.run(x=mca_x, y=mca_data[-1,:])
    # mca_data -= baseline.y

    D = mca_data
    C = np.ones((D.shape[0], 2)) * .5
    C[-1, :] = np.array([1, 0], dtype="float64")

    print("working")
    mca = ca.analysis.mca.MultiComponentAnalysis(
        max_iters=200,
        c_constraints=[ca.analysis.mca.ConstraintNonneg(), ca.analysis.mca.ConstraintConv()],
        st_constraints=[ca.analysis.mca.ConstraintNonneg()],
        tolerance_increase=100
    )
    mca_result = mca.fit(D, C=C, verbose=True)

    conv = conversion(mca_result)
    plot_mca_results(mca_result, mca_x, D, mca_times, conv)
    return np.column_stack((mca_times, conv)), mca_result.C


def mca_2_mask(data: ca.base.SignalArray):
    t_slice = slice(0, -1)
    x_slice = ca.utils.general_math.get_slice(data.x, start=760, end=1900)

    mca_times = data.time_zeroed[t_slice]
    mca_x = data.x[x_slice]
    mca_data = data.data[t_slice, x_slice]

    mask = ca.processing.weigths.Slices(
        [
            ca.utils.general_math.get_slice(mca_x, start=875, end=1100),
            ca.utils.general_math.get_slice(mca_x, start=1350, end=1500),
        ],
    )
    mask = mask.get_mask(mca_x, mca_data[0, :])
    mca_x = mca_x[mask]
    mca_data = mca_data[:, mask]

    # baseline = ca.processing.baselines.AsymmetricLeastSquared()
    # baseline.run(x=mca_x, y=mca_data[-1,:])
    # mca_data -= baseline.y

    D = mca_data
    C = np.ones((D.shape[0], 2)) * .5
    C[-1, :] = np.array([.9999, 0.0001], dtype="float64")

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


def mca_3(data: ca.base.SignalArray):
    t_slice = slice(0, -1)
    x_slice = ca.utils.general_math.get_slice(data.x, start=800, end=2000)

    mca_times = data.time_zeroed[t_slice]
    mca_x = data.x[x_slice]
    mca_data = data.data[t_slice, x_slice]

    # baseline = ca.processing.baselines.AsymmetricLeastSquared()
    # baseline.run(x=mca_x, y=mca_data[-1,:])
    # mca_data -= baseline.y

    D = mca_data

    _, C_old = mca_2(data)
    oil = np.ones(C_old.shape[0]) * 0.01
    C = np.column_stack((C_old,oil))
    # C = np.ones((D.shape[0], 3)) * .5
    # C[-1, :] = np.array([0.2, 0, 0.1], dtype="float64")

    print("working")
    mca = ca.analysis.mca.MultiComponentAnalysis(
        max_iters=200,
        c_constraints=[ca.analysis.mca.ConstraintNonneg(), ca.analysis.mca.ConstraintConv(index=[0,1])],
        st_constraints=[],
        tolerance_increase=100
    )
    mca_result = mca.fit(D, C=C, verbose=True)

    conv = conversion(mca_result)
    plot_mca_results(mca_result, mca_x, D, mca_times, conv)
    return np.column_stack((mca_times, conv))


def main():
    data = ca.ir.IRSignalArray.from_file(
        r"G:\Other computers\My Laptop\post_doc_2022\Data\polymerizations\DW2-8\DW2_8_IR2.feather"
    )
    # data.raw_data = np.flip(data.raw_data, axis=1)
    # data.to_feather(r"G:\Other computers\My Laptop\post_doc_2022\Data\polymerizations\DW2-8\DW2_8_IR2_fix.feather")

    signal = data.get_signal(300)
    fig = ca.ir.plot_signal(signal)
    fig.add_trace(go.Scatter(x=signal.x_raw, y=signal.y_raw))
    fig.write_html("temp.html", auto_open=True)

    # data.processor.add(ca.processing.re_sampling.CutOffValue(x_span=529, cut_off_value=0.01))
    # data.processor.add(ca.processing.translations.ScaleMax(range_=(1700, 1800)))
    # data.processor.add(ca.processing.smoothing.GaussianTime(sigma=1))
    # data.processor.add(ca.processing.smoothing.ExponentialTime(a=0.7))
    # data.processor.add(ca.processing.baselines.Polynomial(
    #         degree=1,
    #         weights=ca.processing.weigths.AdaptiveDistanceMedian(threshold=0.03)
    # ))
    # data.processor.add(ca.processing.smoothing.Gaussian(sigma=2))
    # data.processor.add(ca.processing.translations.ScaleMax(range_=(1700, 1800)))

    # mca_result_1 = mca_2_mask(data)
    # for i in range(mca_result_1.shape[0]):
    #     print(mca_result_1[i, 0], mca_result_1[i, 1])


if __name__ == "__main__":
    main()
