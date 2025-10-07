import re
from datetime import datetime

import numpy as np
import plotly.graph_objs as go


def get_data():
    file_path = r"G:\Other computers\My Laptop\post_doc_2022\Data\polymerizations\DW2-5\DW2-5-1-ATIR.csv"
    data = np.loadtxt(file_path, delimiter=",")

    wavenumber = data[0, 1:]
    times = data[1:, 0]
    data = np.flip(data[1:, 1:], axis=1)

    data = data - np.mean(data[0:10, :], axis=0)
    return times, wavenumber, data


def single():
    times, wavenumber, data = get_data()
    times = times - times[0]
    trace_index = [0, 500, -1]

    fig = go.Figure()
    for i in trace_index:
        fig.add_trace(
            go.Scatter(
                x=wavenumber,
                y=data[i, :],
                mode="lines",
                legendgroup="lines",
                showlegend=True,
                name=f"{i}|{times[i]}"
            )
        )
    fig.write_html("temp_main.html", auto_open=True)


def main():
    times, wavenumber, data = get_data()
    times = times - times[0]
    data = data[150:300:10]
    times = times[150:300:10]
    n = len(times)

    fig = go.Figure()
    # fig.add_trace(
    #     go.Surface(
    #         x=wavenumber,
    #         y=times,
    #         z=data,
    #         legendgroup="surface",
    #         showlegend=True,
    #         showscale=False
    #     )
    # )

    for i in range(n):
        fig.add_trace(
            go.Scatter3d(
                x=wavenumber,
                y=times[i] * np.ones_like(wavenumber),
                z=data[i, :],
                mode="lines",
                line={"color": "black"},
                legendgroup="lines",
                showlegend=False if i != 0 else True,
            )
        )

    fig.update_layout(scene=dict(
        yaxis_title='rxn time (sec)',
        xaxis_title='wavenumber (cm-1)',
        zaxis_title='signal'),
    )

    # fig.update_layout(autosize=False, width=800, height=600, font=dict(family="Arial", size=18, color="black"),
    #                   plot_bgcolor="white", showlegend=False)
    # fig.update_xaxes(title="<b>retention time (min)</b>", tickprefix="<b>", ticksuffix="</b>", showline=True,
    #                  linewidth=5, mirror=True, linecolor='black', ticks="outside", tickwidth=4, showgrid=False,
    #                  gridwidth=1, gridcolor="lightgray", range=[8, 14])
    # fig.update_yaxes(title="<b>normalized signal</b>", tickprefix="<b>", ticksuffix="</b>", showline=True,
    #                  linewidth=5, mirror=True, linecolor='black', ticks="outside", tickwidth=4, showgrid=False,
    #                  gridwidth=1, gridcolor="lightgray", range=[-0.1, 1.1])

    # create gif
    # from plotly_gif import GIF, three_d_scatter_rotate
    # gif = GIF(mode="png")
    # three_d_scatter_rotate(gif, fig)

    fig.write_html("temp_main.html", auto_open=True)


from pymcr.constraints import ConstraintNonneg, Constraint


class ConstraintConv(Constraint):
    """
    Conversion constraint. sum(C) = 1

    Parameters
    ----------
    copy : bool
        Make copy of input data, A; otherwise, overwrite (if mutable)
    """

    def __init__(self, copy=False):
        """ A must be non-negative"""
        super().__init__(copy)

    def transform(self, A):
        """ Apply nonnegative constraint"""
        if self.copy:
            B = np.copy(A)
            for i, row in enumerate(B):
                total = np.sum(row)
                B[i] = B[i] / total
            return B
        else:
            for i, row in enumerate(A):
                total = np.sum(row)
                A[i] = A[i] / total
            return A


def get_wavenumber_index(wavelength: np.ndarray, start: None | int | float = None, end: None | int | float = None) \
    -> slice:
    if start is None:
        start_ = 0
    else:
        start_ = np.argmin(np.abs(wavelength-start))
    if end is None:
        end_ = -1
    else:
        end_ = np.argmin(np.abs(wavelength-end))

    return slice(start_, end_)


def analysis():
    """
    https://www.github.com/usnistgov/pyMCR

    """

    import numpy as np
    import plotly.graph_objs as go
    from pymcr.mcr import McrAR
    from pymcr.regressors import OLS, NNLS
    from pymcr.constraints import ConstraintNonneg, ConstraintNorm

    """
    D = CS^T

    D = row: spectra, column: is time

    C =

    S = 

    """
    times, wavenumber, data = get_data()
    times_ = np.copy(times)

    times = times - times[0]
    t_slice = slice(150, -1)
    w_slice = get_wavenumber_index(wavenumber, start=900, end=None)
    times = times[t_slice]
    times_ = times_[t_slice]
    wavenumber = wavenumber[w_slice]
    data = data[t_slice, w_slice]


    num_compounds = 2
    D = data
    C = np.ones((D.shape[0], num_compounds)) * .5
    C[30, :] = np.array([1, 0])
    # C[20, :] = np.array([0, .5])
    # S = np.ones((2, D.shape[1])) * -.1
    # S[0, :] = data[-1, :]

    mcrar = McrAR(max_iter=200, c_constraints=[ConstraintNonneg(), ConstraintConv()],
                  # c_regr='NNLS', # st_regr='NNLS', #  ConstraintNonneg(),
                  st_constraints=[], tol_increase=3)

    mcrar.fit(D, C=C, verbose=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=normalize(mcrar.ST_[0, :]), x=wavenumber, name="MA"))
    fig.add_trace(go.Scatter(y=normalize(mcrar.ST_[1, :]), x=wavenumber, name="PMA"))
    fig.add_trace(go.Scatter(y=normalize(data[0, :]), x=wavenumber, name="sample early"))
    fig.add_trace(go.Scatter(y=normalize(data[20, :]), x=wavenumber, name="sample middle"))
    fig.add_trace(go.Scatter(y=normalize(data[-1, :]), x=wavenumber, name="sample late"))

    fig.update_layout(autosize=False, width=1200, height=600, font=dict(family="Arial", size=18, color="black"),
                      plot_bgcolor="white", showlegend=True)
    fig.update_xaxes(title="<b>wavenumber (cm-1) (min)</b>", tickprefix="<b>", ticksuffix="</b>", showline=True,
                     linewidth=5, mirror=True, linecolor='black', ticks="outside", tickwidth=4, showgrid=False,
                     gridwidth=1, gridcolor="lightgray", autorange="reversed")
    fig.update_yaxes(title="<b>normalized absorbance</b>", tickprefix="<b>", ticksuffix="</b>", showline=True,
                     linewidth=5, mirror=True, linecolor='black', ticks="outside", tickwidth=4, showgrid=False,
                     gridwidth=1, gridcolor="lightgray")

    fig.write_html("temp.html", auto_open=True)

    conv = mcrar.C_[:, 1] / (mcrar.C_[:, 0] + mcrar.C_[:, 1])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=conv))
    fig.update_layout(autosize=False, width=800, height=600, font=dict(family="Arial", size=18, color="black"),
                      plot_bgcolor="white", showlegend=False)
    fig.update_xaxes(title="<b>rxn time (min)</b>", tickprefix="<b>", ticksuffix="</b>", showline=True,
                     linewidth=5, mirror=True, linecolor='black', ticks="outside", tickwidth=4, showgrid=False,
                     gridwidth=1, gridcolor="lightgray")
    fig.update_yaxes(title="<b>conversion</b>", tickprefix="<b>", ticksuffix="</b>", showline=True,
                     linewidth=5, mirror=True, linecolor='black', ticks="outside", tickwidth=4, showgrid=False,
                     gridwidth=1, gridcolor="lightgray", range=[0, 1])
    fig.write_html("temp2.html", auto_open=True)

    for i in range(mcrar.C_.shape[0]):
        print(datetime.fromtimestamp(times_[i]), conv[i])


def normalize(data):
    return data / np.max(data)


def analysis2():
    from pymcr.mcr import McrAR
    times, wavenumber, data = get_data()
    times_ = np.copy(times)

    times = times - times[0]
    t_start = 25
    t_end = 800
    times = times[t_start:t_end]
    times_ = times_[t_start:t_end]
    data = data[t_start:t_end]

    wave_number_mask = np.where(np.logical_and(wavenumber > 933, 1800 > wavenumber))[0]
    data = data[:, wave_number_mask]
    wavenumber = wavenumber[wave_number_mask]

    D = data
    S = np.ones((2, D.shape[1])) * -.5
    S[0, :] = data[-1, :]
    S[1, :] = data[150, :]

    mcrar = McrAR(max_iter=100, c_constraints=[ConstraintNonneg(), ConstraintConv()], st_regr='NNLS', #c_regr='NNLS',
                  st_constraints=[], tol_increase=3)

    mcrar.fit(D, ST=S, verbose=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=normalize(mcrar.ST_[0, :]), x=wavenumber, name="MA"))
    fig.add_trace(go.Scatter(y=normalize(mcrar.ST_[1, :]), x=wavenumber, name="PMA"))
    fig.add_trace(go.Scatter(y=normalize(data[3, :]), x=wavenumber, name="sample early"))
    fig.add_trace(go.Scatter(y=normalize(data[405, :]), x=wavenumber, name="sample middle"))
    fig.add_trace(go.Scatter(y=normalize(data[-3, :]), x=wavenumber, name="sample late"))

    fig.update_layout(autosize=False, width=1200, height=600, font=dict(family="Arial", size=18, color="black"),
                      plot_bgcolor="white", showlegend=True)
    fig.update_xaxes(title="<b>wavenumber (cm-1) (min)</b>", tickprefix="<b>", ticksuffix="</b>", showline=True,
                     linewidth=5, mirror=True, linecolor='black', ticks="outside", tickwidth=4, showgrid=False,
                     gridwidth=1, gridcolor="lightgray", autorange="reversed")
    fig.update_yaxes(title="<b>normalized absorbance</b>", tickprefix="<b>", ticksuffix="</b>", showline=True,
                     linewidth=5, mirror=True, linecolor='black', ticks="outside", tickwidth=4, showgrid=False,
                     gridwidth=1, gridcolor="lightgray")

    fig.write_html("temp3.html", auto_open=True)

    conv = mcrar.C_[:, 1] / (mcrar.C_[:, 0] + mcrar.C_[:, 1])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=conv))
    fig.update_layout(autosize=False, width=800, height=600, font=dict(family="Arial", size=18, color="black"),
                      plot_bgcolor="white", showlegend=False)
    fig.update_xaxes(title="<b>rxn time (min)</b>", tickprefix="<b>", ticksuffix="</b>", showline=True,
                     linewidth=5, mirror=True, linecolor='black', ticks="outside", tickwidth=4, showgrid=False,
                     gridwidth=1, gridcolor="lightgray")
    fig.update_yaxes(title="<b>conversion</b>", tickprefix="<b>", ticksuffix="</b>", showline=True,
                     linewidth=5, mirror=True, linecolor='black', ticks="outside", tickwidth=4, showgrid=False,
                     gridwidth=1, gridcolor="lightgray", range=[0, 1])
    fig.write_html("temp4.html", auto_open=True)

    for i in range(mcrar.C_.shape[0]):
        print(times[i], conv[i])


if __name__ == "__main__":
    # main()
    single()
    # analysis()
    # analysis2()