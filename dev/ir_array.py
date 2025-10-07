import pathlib
import time

import numpy as np
import plotly.graph_objs as go

import chem_analysis as ca
import chem_analysis.analysis.multi_component_analysis as mca


def main():
    start = time.time()
    file_path = pathlib.Path(r"G:\Other computers\My Laptop\post_doc_2022\Data\polymerizations\DW2-5\DW2-5-1-ATIR.csv")
    data = np.loadtxt(file_path, delimiter=",")
    print(time.time()-start)
    wavenumber = np.flip(data[0, 1:])
    times = data[1:, 0]
    data = data[1:, 1:]

    array_ = ca.ir.IRArray(x=wavenumber, y=times, z=data)
    array_.processor.add(chem_analysis.processing.translations.Subtract(np.mean(array_.z[0:10, :], axis=0)))

    print(time.time() - start)
    fig = ca.ir.plot_signal(array_.get_signal(0))
    print(time.time() - start)
    fig.write_html("temp.html", auto_open=True)
    print(time.time() - start)
    # fig = ca.ir.plot_signal_array_overlap(array_, y_range_index=slice(0, -1, 25))
    # fig.write_html("temp.html", auto_open=True)
    # fig2 = ca.ir.plot_3D_traces(array_, x_range=slice(900, 2000), y_range_index=slice(0, -1, 25))
    # fig2.write_html("temp.html", auto_open=True)

    mca_ = mca.MultiComponentAnalysis(
        c_constraints=[mca.ConstraintNonneg(), mca.ConstraintConv()]
    )
    D = array_.z[150:, :]
    C = np.ones((D.shape[0], 2)) * .5
    C[0, :] = np.array([1, 0])
    results = mca_.fit(D, C, verbose=True)
    print(results.C)

    conv = results.C[:, 1] / (results.C[:, 0] + results.C[:, 1])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=conv))
    fig.write_html("temp1.html", auto_open=True)


if __name__ == "__main__":
    main()
