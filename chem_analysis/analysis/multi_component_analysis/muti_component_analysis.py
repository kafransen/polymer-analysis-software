import logging
from typing import Callable, Iterable

import numpy as np

from chem_analysis.analysis.multi_component_analysis.constraints import Constraint
from chem_analysis.analysis.multi_component_analysis.regressors import LinearRegressor, LeastSquares
from chem_analysis.analysis.multi_component_analysis.metrics import MetricType, mean_square_error

logger = logging.getLogger(__name__)


class MCAResult:
    """

    err: list
        List of calculated errors (from error_function) after each least squares (ie
        twice per iter)
    C_: ndarray [n_samples, n_targets]
        Most recently calculated C matrix (that did not cause a tolerance
        failure)
    ST_: ndarray [n_targets, n_features]
        Most recently calculated S^T matrix (that did not cause a tolerance
        failure)
    C_opt_: ndarray [n_samples, n_targets]
        [Optimal] C matrix for lowest err attribute
    ST_opt_: ndarray [n_targets, n_features]
        [Optimal] ST matrix for lowest err attribute
    total_iters: int
        Total number of iters performed
    optimal_iter: int
        iter when optimal C and ST calculated
    exit_max_iters_reached: bool
        Exited iters due to maximum number of iter reached (max_iters
        parameter)
    exit_tolerance_increase: bool
        Exited iters due to maximum fractional increase in error metric
        (via error_function)
    exit_tolerance_n_increase: bool
        Exited iters due to maximum number of consecutive increases in
        error metric (via err function)
    exit_tolerance_error_change: bool
        Exited iters due to error metric change that is smaller than
        tolerance_error_change
    exit_tolerance_n_above_min: bool
        Exited iters due to maximum number of half-iters for which
        the error metric increased above the minimum error
    """

    def __init__(self):
        self.C = None
        self.ST = None
        self.error = None

        self.optimal_iter = None
        self.total_iters = None
        self.exit_max_iters_reached = False
        self.exit_tolerance_increase = False
        self.exit_tolerance_n_increase = False
        self.exit_tolerance_error_change = False
        self.exit_tolerance_n_above_min = False

    @property
    def D(self):
        """ D matrix with optimal C and S^T matrices """
        return np.dot(self.C, self.ST)


CallbackType = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None]


class MultiComponentAnalysis:
    """
    Multivariate Curve Resolution - Alternating Regression

    D = CS^T

    """

    def __init__(self,
                 c_regressor: LinearRegressor = LeastSquares(),
                 st_regressor: LinearRegressor = LeastSquares(),
                 c_constraints: list[Constraint] = None,
                 st_constraints: list[Constraint] = None,
                 max_iters: int = 50,
                 error_function: MetricType = mean_square_error,
                 tolerance_increase: float = 0.0,
                 tolerance_error_change: float = None,
                 iters_above_min: int = 10
                 ):
        """
        Parameters
        ----------
        c_regressor: LinearRegressor
            Regressor for calculating the C matrix
        st_regressor: LinearRegressor
            Regressor for calculating the S^T matrix
        c_constraints: list
            List of constraints applied to calculation of C matrix
        st_constraints: list
            List of constraints applied to calculation of S^T matrix
        max_iters: int
            Maximum number of iters. One iter calculates both C and S^T
        error_function: 
            Function to calculate error/differences after each least squares
            calculation (ie twice per iter). Outputs to err attribute.
        tolerance_increase: float
            Factor increases to allow in err attribute. Set to 0 for no increase
            allowed. E.g., setting to 1.0 means the err can double per iter.
        tolerance_error_change: float
            If err changes less than tolerance_error_change, per iter, break.
        iters_above_min: int
            Number of half-iters that can be performed without reaching a
            new error-minimu

        Notes
        -----
        -   Setting any tolerance to None turns that check off

        """
        self.max_iters = max_iters
        self.tolerance_increase = abs(tolerance_increase)
        self.tolerance_error_change = abs(tolerance_error_change) if tolerance_error_change is not None else None
        self.iters_above_min = iters_above_min

        self.error_function = error_function

        self.c_constraints = c_constraints if c_constraints is not None else []
        self.st_constraints = st_constraints if st_constraints is not None else []
        self.c_regressor = c_regressor
        self.st_regressor = st_regressor

    def fit(self,
            D: np.ndarray,
            C: np.ndarray = None,
            ST: np.ndarray = None,
            st_fix: Iterable[int] = None,
            c_fix: Iterable[int] = None,
            c_first: bool = True,
            verbose: bool = False,
            callback: CallbackType = None,
            half_step_callback: CallbackType = None
            ):
        """
        Perform MCR-AR. D = CS^T. Solve for C and S^T iteratively.

        Parameters
        ----------
        D: 
            D matrix
        C: 
            Initial C matrix estimate. Only provide initial C OR S^T.
        ST: 
            Initial S^T matrix estimate. Only provide initial C OR S^T.
        st_fix: 
            The spectral component numbers to keep fixed.
        c_fix: 
            The concentration component numbers to keep fixed.
        c_first: 
            Calculate C first when both C and ST are provided. c_fix and st_fix
            must also be provided in this circumstance.
        verbose: 
            Log iter_ and per-least squares err results. See Notes.
        callback: 
            Function to perform after each iter_
        half_step_callback: 
            Function to perform after half-iter_

        """
        logger.setLevel(logging.DEBUG) if verbose else logger.setLevel(logging.INFO)

        # Ensure only C or ST provided
        if C is None and (ST is None):
            raise TypeError('C or ST estimate must be provided')
        if C is not None and ST is not None and (c_fix is None or st_fix is None):
            raise TypeError('Only C or ST estimate must be provided unless c_fix and st_fix are both provided')

        C = np.asanyarray(C) if C is not None else C
        ST = np.asanyarray(ST) if ST is not None else ST
        D = np.asanyarray(D)

        # Both C and ST provided. special_skip_c comes into play below
        both_condition = ST is not None and C is not None and not c_first

        result = MCAResult()
        result.error = np.zeros(self.max_iters, dtype=np.float64)
        iters_with_increase = 0
        for iter_ in range(self.max_iters):
            # Both st and c provided, but c_first is False
            special_skip_c = True if iter_ == 0 and both_condition else False

            if ST is not None and not special_skip_c:
                C_temp = self.c_regressor.fit(ST.T, D.T)

                # Apply fixed C's
                if c_fix:
                    C_temp[:, c_fix] = C[:, c_fix]

                # Apply c-constraints
                for constraint in self.c_constraints:
                    C_temp = constraint.transform(C_temp)

                # Apply fixed C's
                if c_fix:
                    C_temp[:, c_fix] = C[:, c_fix]

                D_calc = np.dot(C_temp, ST)

                if self._check_stopping_half_iter(result, D, D_calc, C_temp, ST, iter_, iters_with_increase):
                    break
                C = C_temp  # self.ST_ = 1 * ST_temp

                if half_step_callback is not None:
                    half_step_callback(C, ST, D, D_calc)

            if C is not None:
                ST_temp = self.st_regressor.fit(C, D).T

                # Apply fixed ST's
                if st_fix:
                    ST_temp[st_fix] = ST[st_fix]

                # Apply ST-constraints
                for constraint in self.st_constraints:
                    ST_temp = constraint.transform(ST_temp.T).T

                # Apply fixed ST's
                if st_fix:
                    ST_temp[st_fix] = ST[st_fix]

                D_calc = np.dot(C, ST_temp)

                if self._check_stopping_half_iter(result, D, D_calc, C, ST_temp, iter_, iters_with_increase):
                    break
                ST = ST_temp

                if callback is not None:
                    callback(C, ST, D, D_calc)

            if self._check_stopping(iter_, result):
                return result
        else:
            logger.info('Max iters reached ({}).'.format(self.max_iters + 1))
            result.exit_max_iters_reached = True

        result.total_iters = iter_
        return result

    def _check_stopping(self, current_error_index: int, result: MCAResult):
        # Check if err changed (absolute value), per iter, less than abs(tolerance_error_change)
        if self.tolerance_error_change is not None and current_error_index > 2:
            error_differ = np.abs(result.error[current_error_index - 1] - result.error[current_error_index - 3])
            if error_differ < self.tolerance_error_change:
                logger.info(f'Change in err below tolerance_error_change({error_differ:.4e}). Exiting.')
                result.exit_tolerance_error_change = True
                return True

        return False

    def _check_stopping_half_iter(self,
                                  result: MCAResult,
                                  D: np.ndarray,
                                  D_calc: np.ndarray,
                                  C: np.ndarray,
                                  ST: np.ndarray,
                                  iter_: int,
                                  iters_with_increase: int
                                  ) -> bool:
        # calculate error
        error = self.error_function(D, D_calc)
        result.error[iter_] = error

        # check for tolerance increase
        if iter_ == 0 or error < np.min(result.error[:iter_]):
            result.C = C
            result.ST = ST
            result.optimal_iter = iter_
            iters_with_increase = 0
        else:
            iters_with_increase += 1

        if iter_ == 0:
            return False

        # iter_above_min
        if self.iters_above_min is not None and iters_with_increase > self.iters_above_min:
            logger.info(f'Stop: Error increased for {iters_with_increase} times.')
            result.exit_iters_above_min = True
            return True

        # tolerance_increase
        if self.tolerance_increase is not None:
            if error > result.error[iter_ - 1] * (1 + self.tolerance_increase):
                logger.info('Stop: Error increased above tolerance.')
                result.exit_tolerance_increase = True
                return True

        return False
