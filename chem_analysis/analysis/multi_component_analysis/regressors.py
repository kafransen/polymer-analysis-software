import abc

import numpy as np
from scipy.linalg import lstsq
from scipy.optimize import nnls


class LinearRegressor(abc.ABC):
    """ Abstract class for linear regression methods """
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @abc.abstractmethod
    def fit(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """ AX = B, solve for X """


class LeastSquares(LinearRegressor):
    """
    Ordinary least squares regression

    AX = B, solve for X (coefficients.T)

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html

    """
    def fit(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """ AX = B, solve for X """
        x, _, _, _ = lstsq(A, b, **self.kwargs)

        if x is not None:
            """ The transposed form of X. This is the formalism of scikit-learn """
            x = x.T  # The transposed form of X. This is the formalism of scikit-learn

        return x


class NonNegativeLeastSquares(LinearRegressor):
    """
    Non-negative constrained least squares regression

    argmin_x || Ax - b ||_2 for x>=0

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.nnls.html

    """

    def fit(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """ AX = B, solve for X """
        if b.ndim == 2:
            n = b.shape[-1]
            x = np.zeros((A.shape[-1], n))
            for num in range(n):
                x[:, num], _ = nnls(A, b[:, num], **self.kwargs)
        else:
            x, _ = nnls(A, b, **self.kwargs)

        if x is not None:
            x = x.T  # The transposed form of X. This is the formalism of scikit-learn

        return x
