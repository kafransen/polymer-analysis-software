from typing import Iterable

import numpy as np
from scipy import sparse

from chem_analysis.utils.math import MIN_FLOAT
from chem_analysis.processing.weigths.weights import DataWeight
import chem_analysis.utils.validation as validation
from chem_analysis.processing.baselines.base import BaselineCorrection

diagonals = [
        [-1, 1],
        [1, -2, 1],
        [-1, 3, -3, 1],
    ]


def asymmetric_least_squared(
        y: np.ndarray,
        lambda_: float = 1e6,
        p: float = 1e-2,
        diff_order: int = 2,
        max_iter: int = 50,
        tol: float = 1e-3,
        weights: np.ndarray = None
) -> tuple[np.ndarray, dict]:
    """
   Asymmetric least squared (AsLS) fitting.

    Parameters
    ----------
    y:
        y data
    lambda_:
        smoothing parameter
        larger values = smoother baselines
    p:
        penalizing weighting factor
        0 < p < 1
    diff_order:
        values: 1, 2, 3
       order of the differential matrix
    max_iter:
        max number of fit iterations
    tol:
        error tolerance for termination
    weights:
        weights

    Returns
    -------
    baseline:
    params :

    """
    # check inputs
    if max_iter < 2:
        raise ValueError("max_iter needs to be greater than 2")
    if p < 0 or p > 1:
        raise ValueError('p must be between 0 and 1')
    if diff_order not in (1, 2, 3):
        raise ValueError(f'diff_order must be 1,2,3. \n\tgiven: {diff_order}')
    if weights is None:
        weight_array = np.ones(y.shape[0])
    else:
        validation.check_array_size(weights, y.shape, "asymmetric_least_squared.weights")
        validation.check_array_inf_nan(weights, "asymmetric_least_squared.weights")
        weight_array = np.asarray(weights).copy()

    # setup
    diff_matrix = sparse.diags(
        diagonals[diff_order-1],
        list(range(diff_order + 1)),
        shape=(y.shape[0] - diff_order, y.shape[0])
    )
    d = lambda_ * diff_matrix.T * diff_matrix
    w = sparse.diags(weight_array)

    # solve
    tolerances = np.empty(max_iter + 1)
    for i in range(max_iter + 1):
        baseline = sparse.linalg.spsolve(w + d, weight_array * y)

        mask = y > baseline
        new_weights = p * mask + (1 - p) * np.logical_not(mask)

        rel_difference = np.sum(np.abs(weight_array - new_weights)) / np.sum(np.abs(weight_array + new_weights))
        tolerances[i] = rel_difference
        if rel_difference < tol:
            break
        weight_array = new_weights
        w.setdiag(weight_array)

    params = {'weights': weight_array, 'tolerances': tolerances[:i + 1]}
    return baseline, params


class AsymmetricLeastSquared(BaselineCorrection):
    def __init__(self,
                 lambda_=1e6,
                 p=1e-2,
                 diff_order=2,
                 max_iter=50,
                 tol=1e-3,
                 weights: DataWeight | Iterable[DataWeight] = None
                 ):
        super().__init__(weights)
        self.lambda_ = lambda_
        self.p = p
        self.diff_order = diff_order
        self.max_iter = max_iter
        self.tol = tol

    def get_baseline(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.weights is not None:
            mask = self.weights.get_mask(x, y)
            x_ = x[mask]
            y_ = y[mask]
        else:
            x_ = x
            y_ = y

        y_baseline, params = asymmetric_least_squared(
            y_,
            self.lambda_,
            self.p,
            self.diff_order,
            self.max_iter,
            self.tol,
        )
        return np.interp(x, x_, y_baseline)


def improved_asymmetric_least_squared(
        y: np.ndarray,
        lambda_: float = 1e6,
        lambda_1: float = 1e-4,
        p: float = 1e-2,
        diff_order: int = 2,
        max_iter: int = 50,
        tol: float = 1e-3,
        weights: np.ndarray = None
):
    """
   Asymmetric least squared (AsLS) fitting.

    Parameters
    ----------
    y:
        y data
    lambda_:
        smoothing parameter
        larger values = smoother baselines
    p:
        penalizing weighting factor
        0 < p < 1
    diff_order:
        values: 1, 2, 3
       order of the differential matrix
    max_iter:
        max number of fit iterations
    tol:
        error tolerance for termination
    weights:
        weights

    Returns
    -------
    baseline:
    params :

    """
    # check inputs
    if max_iter < 2:
        raise ValueError("max_iter needs to be greater than 2")
    if p < 0 or p > 1:
        raise ValueError('p must be between 0 and 1')
    if diff_order not in (1, 2, 3):
        raise ValueError(f'diff_order must be 1,2,3. \n\tgiven: {diff_order}')
    if weights is None:
        weight_array = np.ones(y.shape[0])
    else:
        validation.check_array_size(weights, y.shape, "asymmetric_least_squared.weights")
        validation.check_array_inf_nan(weights, "asymmetric_least_squared.weights")
        weight_array = np.asarray(weights).copy()

    # setup
    diff_matrix = sparse.diags(
        diagonals[diff_order-1],
        list(range(diff_order + 1)),
        shape=(y.shape[0] - diff_order, y.shape[0])
    )
    d = lambda_ * diff_matrix.T * diff_matrix
    w = sparse.diags(weight_array)

    # solve
    d1_y = y.copy()
    d1_y[0] = y[0] - y[1]
    d1_y[-1] = y[-1] - y[-2]
    d1_y[1:-1] = 2 * y[1:-1] - y[:-2] - y[2:]
    d1_y = lambda_1 * d1_y
    tolerances = np.empty(max_iter + 1)
    for i in range(max_iter + 1):
        weight_squared = weight_array * weight_array
        baseline = sparse.linalg.spsolve(w + d, weight_squared * y + d1_y)

        mask = y > baseline
        new_weights = p * mask + (1 - p) * np.logical_not(mask)

        rel_difference = np.sum(np.abs(weight_array - new_weights)) / np.sum(np.abs(weight_array + new_weights))
        tolerances[i] = rel_difference
        if rel_difference < tol:
            break
        weight_array = new_weights
        w.setdiag(weight_array)

    params = {'weights': weight_array, 'tolerances': tolerances[:i + 1]}
    return baseline, params


class ImprovedAsymmetricLeastSquared(BaselineCorrection):
    def __init__(self,
                 lambda_: float = 1e6,
                 lambda_1: float = 1e-4,
                 p: float = 1e-2,
                 diff_order: int = 2,
                 max_iter: int = 50,
                 tol: float = 1e-3,
                 weights: DataWeight | Iterable[DataWeight] = None
                 ):
        super().__init__(weights)
        self.lambda_ = lambda_
        self.lambda_1 = lambda_1
        self.p = p
        self.diff_order = diff_order
        self.max_iter = max_iter
        self.tol = tol
        self._weights = weights

    def get_baseline(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.weights is not None:
            mask = self.weights.get_mask(x, y)
            x_ = x[mask]
            y_ = y[mask]
        else:
            x_ = x
            y_ = y

        y_baseline, params = improved_asymmetric_least_squared(
            y_,
            self.lambda_,
            self.lambda_1,
            self.p,
            self.diff_order,
            self.max_iter,
            self.tol
        )
        return np.interp(x, x_, y_baseline)


def reweighted_improved_asymmetric_least_squared(
        y: np.ndarray,
        lambda_: float = 1e6,
        diff_order: int = 2,
        max_iter: int = 50,
        tol: float = 1e-3,
        weights: np.ndarray = None
):
    """
   Improved reweighted Asymmetric least squared (AsLS) fitting.

    Parameters
    ----------
    y:
        y data
    lambda_:
        smoothing parameter
        larger values = smoother baselines
    diff_order:
        values: 1, 2, 3
       order of the differential matrix
    max_iter:
        max number of fit iterations
    tol:
        error tolerance for termination
    weights:
        weights

    Returns
    -------
    baseline:
    params :

    """
    # check inputs
    if max_iter < 2:
        raise ValueError("max_iter needs to be greater than 2")
    if diff_order not in (1, 2, 3):
        raise ValueError(f'diff_order must be 1,2,3. \n\tgiven: {diff_order}')
    if weights is None:
        weight_array = np.ones(y.shape[0])
    else:
        validation.check_array_size(weights, y.shape, "reweighted_improved_asymmetric_least_squared.weights")
        validation.check_array_inf_nan(weights, "reweighted_improved_asymmetric_least_squared.weights")
        weight_array = np.asarray(weights).copy()

    # setup
    diff_matrix = sparse.diags(
        diagonals[diff_order-1],
        list(range(diff_order + 1)),
        shape=(y.shape[0] - diff_order, y.shape[0])
    )
    d = lambda_ * diff_matrix.T * diff_matrix
    w = sparse.diags(weight_array)

    # solve
    tolerances = np.empty(max_iter + 1)
    for i in range(max_iter + 1):
        baseline = sparse.linalg.spsolve(w + d, weight_array * y)

        residual = y - baseline
        std = np.std(residual[residual < 0], ddof=1)
        if std == 0:
            std = MIN_FLOAT
        inner = (np.exp(i-1) / std) * (residual - 2 * std)
        new_weights = 0.5 * (1 - (inner / np.sqrt(1 + inner ** 2)))

        rel_difference = np.sum(np.abs(weight_array - new_weights)) / np.sum(np.abs(weight_array + new_weights))
        tolerances[i] = rel_difference
        if not np.isfinite(rel_difference):
            break
        if rel_difference < tol:
            break
        weight_array = new_weights
        w.setdiag(weight_array)

    params = {'weights': weight_array, 'tolerances': tolerances[:i + 1]}

    return baseline, params


class ReweightedImprovedAsymmetricLeastSquared(BaselineCorrection):
    def __init__(self,
                 lambda_=1e6,
                 diff_order=2,
                 max_iter=50,
                 tol=1e-3,
                 weights: DataWeight | Iterable[DataWeight] = None
                 ):
        super().__init__(weights)
        self.lambda_ = lambda_
        self.diff_order = diff_order
        self.max_iter = max_iter
        self.tol = tol
        self._weights = weights

    def get_baseline(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.weights is not None:
            mask = self.weights.get_mask(x, y)
            x_ = x[mask]
            y_ = y[mask]
        else:
            x_ = x
            y_ = y

        y_baseline, params = reweighted_improved_asymmetric_least_squared(
            y_,
            self.lambda_,
            self.diff_order,
            self.max_iter,
            self.tol
        )
        return np.interp(x, x_, y_baseline)


def adaptive_asymmetric_least_squared(
        y: np.ndarray,
        lambda_: float = 1e6,
        diff_order: int = 2,
        max_iter: int = 50,
        tol: float = 1e-3,
        weights: np.ndarray = None
) -> tuple[np.ndarray, dict]:
    """
   adaptive iteratively reweighted penalized least squares

    Parameters
    ----------
    y:
        y data
    lambda_:
        smoothing parameter
        larger values = smoother baselines
    p:
        penalizing weighting factor
        0 < p < 1
    diff_order:
        values: 1, 2, 3
       order of the differential matrix
    max_iter:
        max number of fit iterations
    tol:
        error tolerance for termination
    weights:
        weights

    Returns
    -------
    baseline:
    params :

    References
    ----------
    Baseline correction using adaptive iteratively reweighted penalized least squares.
    Analyst, 2010, 135(5), 1138-1146.
    DOI: https://doi.org/10.1039/B922045C

    """
    # check inputs
    if max_iter < 2:
        raise ValueError("max_iter needs to be greater than 2")
    if diff_order not in (1, 2, 3):
        raise ValueError(f'diff_order must be 1,2,3. \n\tgiven: {diff_order}')
    if weights is None:
        weight_array = np.ones(y.shape[0])
    else:
        validation.check_array_size(weights, y.shape, "adaptive_asymmetric_least_squared.weights")
        validation.check_array_inf_nan(weights, "adaptive_asymmetric_least_squared.weights")
        weight_array = np.asarray(weights).copy()

    # setup
    diff_matrix = sparse.diags(
        diagonals[diff_order-1],
        list(range(diff_order + 1)),
        shape=(y.shape[0] - diff_order, y.shape[0])
    )
    d = lambda_ * diff_matrix.T * diff_matrix
    w = sparse.diags(weight_array)

    # solve
    y_l1_norm = np.abs(y).sum()
    tolerances = np.empty(max_iter + 1)
    for i in range(max_iter + 1):
        baseline = sparse.linalg.spsolve(w + d, weight_array * y)

        residual = y - baseline
        neg_mask = (residual < 0)
        # same as abs(residual[neg_mask]).sum() since residual[neg_mask] are all negative
        residual_l1_norm = -1 * residual[neg_mask].sum()
        if residual_l1_norm / y_l1_norm < tol:
            break
        weight_array = np.exp(i * abs(residual) / residual_l1_norm) * neg_mask
        w.setdiag(weight_array)

    params = {'weights': weight_array, 'tolerances': tolerances[:i + 1]}
    return baseline, params


class AdaptiveAsymmetricLeastSquared(BaselineCorrection):
    def __init__(self,
                 lambda_=1e6,
                 diff_order=2,
                 max_iter=50,
                 tol=1e-3,
                 weights: DataWeight | Iterable[DataWeight] = None
                 ):
        super().__init__(weights)
        self.lambda_ = lambda_
        self.diff_order = diff_order
        self.max_iter = max_iter
        self.tol = tol

    def get_baseline(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.weights is not None:
            mask = self.weights.get_mask(x, y)
            x_ = x[mask]
            y_ = y[mask]
        else:
            x_ = x
            y_ = y

        y_baseline, params = adaptive_asymmetric_least_squared(
            y_,
            self.lambda_,
            self.diff_order,
            self.max_iter,
            self.tol
        )
        return np.interp(x, x_, y_baseline)
