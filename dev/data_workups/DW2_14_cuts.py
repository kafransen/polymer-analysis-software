import pathlib

import numpy as np
import plotly.graph_objs as go
from scipy.ndimage import gaussian_filter

import chem_analysis as ca


def main():
    remove = [
        [-1269, 0],
        [6745, 7345],
        [14_047, 14_616],
        [21_337, 21_888],
        [28_288, 28_540]
    ]

    path = pathlib.Path(r"C:\Users\nicep\Desktop\Book1.csv")
    data = np.loadtxt(path, delimiter=",")
    print(data.shape, data[-1, 0])

    for r in reversed(remove):  # start from back to front since we are shifting times
        slice_ = ca.utils.math.get_slice(data[:, 0], *r, strict_bounds=False, start_bound=True, end_bound=False)
        if not (slice_.start is None or slice_.stop is None):
            data = np.delete(data, slice_, axis=0)

        # shift times
        if r[1] <= 0:
            mask = data[:, 0] < r[1]
            data[mask, 0] += (r[1] - r[0])
        else:
            mask = data[:, 0] > r[0]
            data[mask, 0] -= (r[1] - r[0])

    print(data.shape, data[-1, 0], np.sum(list(r[1]-r[0] for r in remove)))
    path = path.with_stem(path.stem + "_proc")
    np.savetxt(path, data, delimiter=",")


if __name__ == "__main__":
    main()
