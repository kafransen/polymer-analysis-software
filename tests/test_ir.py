import pathlib

import numpy as np

import chem_analysis as ca


def get_data(path: pathlib.Path):
    data = np.loadtxt(path, delimiter=",")
    wavenumber = np.flip(data[0, 1:])
    times = data[1:, 0]
    data = data[1:, 1:]
    return times, wavenumber, data


def main():
    file_path = pathlib.Path(r"G:\Other computers\My Laptop\post_doc_2022\Data\polymerizations\DW2-4\DW2-4-ATIR.csv")
    times, wavenumber, data = get_data(file_path)

    array = ca.ir.IRArray(x=wavenumber, y=times, z=data)

    fig = ca.ir.plot_ir_array_overlap(array)
    fig.write_html("temp.html", auto_open=True)

    array.processor.add(chem_analysis.processing.baseline_correction.Polynomial(degree=3))

    # array.analysis.add(ca.analysis.peak_picking.)

    C = np.ones((D.shape[0], num_compounds)) * .5
    C[0, :] = np.array([1, 0, 0])
    C[20, :] = np.array([0, .5, .8])
    # S = np.ones((2, D.shape[1])) * -.1
    # S[0, :] = data[-1, :]

    mcrar_results = ca.algorithms.analysis.multi_component_analysis.mcrar(
        D=array.z,
        C=C,
        verbose=True,
        max_iter = 500,
        c_constraints = [ConstraintNonneg(), ConstraintConv()],
        # c_regr='NNLS', # st_regr='NNLS',
        st_constraints=[],
        tol_increase=3
    )

    mcrar_results

if __name__ == "__main__":
    main()
