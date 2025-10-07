import pathlib

import numpy as np
import plotly.graph_objs as go
from scipy.ndimage import gaussian_filter

import chem_analysis as ca


def main():
    remove = [
        [-269, 0],
        [6010, 6416],
        [12_355, 12_925],
        [20_299, 20_895],
    ]

    path = pathlib.Path(r"C:\Users\nicep\Desktop\Book1.csv")
    data = np.loadtxt(path, delimiter=",")
    print(data.shape)

    for r in reversed(remove):  # start from back to front since we are shifting times
        slice_ = ca.utils.general_math.get_slice(data[:, 0], *r)
        data = np.delete(data, slice_, axis=0)
        data[slice_.start:, 0] -= (r[1] - r[0])  # shift time so gap doesn't exist

    print(data.shape)
    path = path.with_stem(path.stem + "_proc")
    np.savetxt(path, data, delimiter=",")


if __name__ == "__main__":
    main()
