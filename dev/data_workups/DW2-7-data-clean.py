import pathlib

import numpy as np
import plotly.graph_objs as go
from scipy.ndimage import gaussian_filter

import chem_analysis as ca


def main():
    remove = [
        [-338, 0],
        [5821, 6183],
        [12_824, 14_308],
        [20_960, 21_780],
        [29_272, 29_325],
        [30_584, 30_728],
    ]

    path = pathlib.Path(r"G:\Other computers\My Laptop\post_doc_2022\Data\polymerizations\DW2-7\sec.csv")
    data = np.loadtxt(path, delimiter=",")
    print(data.shape)

    for r in reversed(remove):  # start from back to front since we are shifting times
        slice_ = ca.utils.general_math.get_slice(data[:, 0], *r)
        data = np.delete(data, slice_, axis=0)
        data[slice_.start:, 0] -= (r[1] - r[0])  # shift time so gap doesn't exist

    print(data.shape)
    path = path.with_stem(path.stem + "_proc")
    np.savetxt(path, data, delimiter=",")


def main2():
    path = pathlib.Path(r"G:\Other computers\My Laptop\post_doc_2022\Data\polymerizations\DW2-7\ir.csv")
    data = np.loadtxt(path, delimiter=",")
    print(data.shape)

    y_ =data[:, 1]
    y = gaussian_filter(y_, 10)
    z = np.ones_like(y)
    a=0.8
    for row in range(1, data.shape[0]):
        z[row] = a * z[row - 1] + (1-a) * y_[row]

    fig = go.Figure(go.Scatter(x=data[:, 0], y=data[:, 1]))
    fig.add_trace(go.Scatter(x=data[:, 0], y=y))
    fig.add_trace(go.Scatter(x=data[:, 0], y=z))
    fig.write_html("temp.html", auto_open=True)


if __name__ == "__main__":
    main2()
