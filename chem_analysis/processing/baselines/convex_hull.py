from typing import Iterable

import numpy as np
from scipy.spatial import ConvexHull

from chem_analysis.processing.baselines.base import BaselineCorrection
from chem_analysis.processing.weigths.weights import DataWeight


def convex_hull_removal(U, wavelengths):
    """
     Performs spectral normalization via convex hull removal

    Parameters
    ----------
    U(p x q)
    wavelengths (p x 1)

    Returns
    -------
    (p x q)

    Reference
    ---------
    Clark, R.N. and T.L. Roush (1984) Reflectance Spectroscopy: Quantitative
    Analysis Techniques for Remote Sensing Applications, J. Geophys. Res., 89, 6329-6340.

    """
    # Metadata and formatting
    wavelengths = np.reshape(wavelengths, (-1, 1))
    p = len(wavelengths)
    q = U.shape[1]
    U = U.T

    U[:, 0] = 0
    U[:, 419] = 0

    normalizedU = np.zeros((420, q))

    # The algorithm
    for s in range(q):
        rifl = U[s, :]
        points = np.vstack((wavelengths, rifl)).T
        hull = ConvexHull(points)
        c = points[hull.vertices]
        d = c[c[:, 1].argsort()]

        xs = d[:, 1]
        ys = d[:, 0]
        unique_indices = np.unique(xs, return_index=True)[1]
        xsp = xs[unique_indices]
        ysp = ys[unique_indices]
        rifl_i = np.interp(wavelengths, xsp, ysp)

        for t in range(420):
            if rifl_i[t] != 0:
                normalizedU[t, s] = rifl[t] / rifl_i[t]
            else:
                normalizedU[t, s] = 1

    return normalizedU.T


class ConvexHull(BaselineCorrection):
    def __init__(self, degree: int = 1, weights: DataWeight | Iterable[DataWeight] = None):
        super().__init__(weights)
        self.degree = degree

    def run(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        weights = self.get_weights(x, y)
        params = np.polyfit(x, y, self.degree, w=weights)
        func_baseline = np.poly1d(params)
        y_baseline = func_baseline(x)
        y = y - y_baseline

        self._y = y_baseline
        self._x = x
        return x, y
