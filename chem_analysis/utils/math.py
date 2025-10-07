import logging

import numpy as np

MIN_FLOAT = np.finfo(float).eps


def pack_time_series(x: np.ndarray, time_: np.ndarray, z: np.array) -> np.ndarray:
    if x.shape[0] != z.shape[1]:
        raise ValueError(f"'x.shape[0]' must equal 'z.shape[1]'\n\tx shape:{x.shape}\n\tz shape:{z.shape}")
    if time_.shape[0] != z.shape[0]:
        raise ValueError(f"'time_.shape[0]' must equal 'z.shape[0]'\n\ttime shape:{time_.shape}\n\tz shape:{z.shape}")

    data = np.empty((len(time_) + 1, len(x) + 1), dtype=z.dtype)
    data[0, 0] = 0
    data[0, 1:] = x
    data[1:, 0] = time_
    data[1:, 1:] = z
    return data


def unpack_time_series(data: np.array) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = data[0, 1:]
    time_ = data[1:, 0]
    z = data[1:, 1:]
    return x, time_, z


def check_for_flip(x: np.ndarray, y: np.ndarray):
    """flip data if giving backwards; it should be low to high"""
    if x[0] > x[-1]:
        x = np.flip(x)
        y = np.flip(y)
    return x, y


def quick_check_for_sorted_array(x: np.ndarray, min_check: int = 5000) -> bool:
    """
    ** Not a strict check ** but it is quick to compute for any size array
    Parameters
    ----------
    x
    min_check

    Returns
    -------

    """
    if len(x) < 2:
        return True
    if x[0] > x[-1]:
        return False

    if len(x) < min_check:
        return np.all(np.all(x[:-1] <= x[1:]))

    i = np.random.randint(1, len(x)-1, min_check-2)
    i.sort()
    return np.all(x[i[:-1]] <= x[i[1:]])


def get_slice(
        x: np.ndarray,
        start=None,
        end=None,
        *,
        checks: bool = True,
        strict_bounds: bool = True,
        start_bound: bool | None = None,
        end_bound: bool | None = None,
) -> slice:
    """
    gets slice from the nearest values

    Parameters
    ----------
    x
        sorted list small -> big
    start:
        value to start slice
    end:
        value to end slice
    checks:
        checks for issues with inputs
    strict_bounds:
        True: raises ValueError if bounds can't be satisfied
        False: puts None in slice
    start_bound:
        None: closest
        False: lowest
        True: highest
    end_bound:
        None: closest
        False: lowest
        True: highest
    Returns
    -------

    """
    if start is None and end is None:
        return slice(None, None)

    if checks:
        if start is not None and end is not None and start > end:
            raise ValueError("'start' value is larger than 'end'. \nFix: Flip bounds.")
        if not quick_check_for_sorted_array(x):
            raise ValueError("Array is not sorted. \nFix: sort 'x'")

    if start is None:
        start_ = None
    elif start_bound is None:
        start_ = np.argmin(np.abs(x - start))
    else:
        if start_bound:
            if end is None:
                mask = (x >= start)
            else:
                mask = (end >= x) & (x >= start)
        else:
            mask = (x <= start)
        if np.all(mask == 0):
            if strict_bounds:
                raise ValueError("slice can't find value for start.")
            else:
                start_ = None
        else:
            start_ = np.argmin(np.abs(x[mask] - start)) + np.argmax(mask)

    if end is None:
        end_ = None
    elif end_bound is None:
        end_ = np.argmin(np.abs(x - end))
    else:
        if end_bound:
            mask = (x >= end)
        else:
            mask = (x <= end) & (x >= start)
        if np.all(mask == 0):
            if strict_bounds:
                raise ValueError("slice can't find value for start.")
            else:
                end_ = None
        else:
            end_ = np.argmin(np.abs(x[mask] - end)) + np.argmax(mask) + 1

    return slice(start_, end_)


def map_argmax_to_original(index: int | np.ndarray, mask) -> int | np.ndarray:
    """
    Map the index from masked array back to the index of the original array.

    Parameters:
    - argmax_index: Index obtained from the argmax of the masked array.
    - mask: Boolean mask indicating which elements were retained in the original array.

    Returns:
    - original_index: Index in the original array corresponding to the argmax of the masked array.
    """
    masked_indices = np.where(mask)[0]  # Get the indices of the retained elements
    original_index = masked_indices[index]
    return original_index


def normalize_by_max(y: np.ndarray) -> np.ndarray:
    max_ = np.max(y)
    # if max_ == 0:
    #     return np.zeros_like(y)
    return y / max_


def normalize_by_area(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x, y = check_for_flip(x, y)
    return y / np.trapz(x=x, y=y)


# pdf = probability distribution function
def get_mean_of_pdf(x: np.ndarray, y: np.ndarray = None, *, y_norm: np.ndarray = None) -> float:
    x, y = check_for_flip(x, y)
    if y_norm is None:
        y_norm = normalize_by_area(x, y)

    return np.trapz(x=x, y=x * y_norm)


def get_standard_deviation_of_pdf(x: np.ndarray, y: np.ndarray = None, *,
                                  y_norm: np.ndarray, mean: int | float = None
                                  ) -> float:
    x, y = check_for_flip(x, y)
    if y_norm is None:
        y_norm = normalize_by_area(x, y)
    if mean is None:
        mean = get_mean_of_pdf(x, y_norm=y_norm)

    return np.sqrt(np.trapz(x=x, y=y_norm * (x - mean) ** 2))


def get_skew_of_pdf(x: np.ndarray, y: np.ndarray = None, *,
                    y_norm: np.ndarray = None,
                    mean: int | float = None,
                    standard_deviation: int | float = None
                    ) -> float:
    x, y = check_for_flip(x, y)
    if y_norm is None:
        y_norm = normalize_by_area(x, y)
    if mean is None:
        mean = get_mean_of_pdf(x, y_norm=y_norm)
    if standard_deviation is None:
        standard_deviation = get_standard_deviation_of_pdf(x, y_norm=y_norm, mean=mean)

    return np.trapz(x=x, y=y_norm * (x - mean) ** 3) / standard_deviation ** 3


def get_kurtosis_of_pdf(x: np.ndarray, y: np.ndarray = None, *,
                        y_norm: np.ndarray = None,
                        mean: int | float = None,
                        standard_deviation: int | float = None
                        ) -> float:
    x, y = check_for_flip(x, y)
    if y_norm is None:
        y_norm = normalize_by_area(x, y)
    if mean is None:
        mean = get_mean_of_pdf(x, y_norm=y_norm)
    if standard_deviation is None:
        standard_deviation = get_standard_deviation_of_pdf(x, y_norm=y_norm, mean=mean)

    return (np.trapz(x=x, y=y_norm * (x - mean) ** 4) / standard_deviation ** 4) - 3


def get_full_width_at_height(x: np.ndarray, y: np.ndarray, height: float | int = 0.5) -> float:
    """ Calculates full width at a height. """
    lower, high = get_width_at(x, y, height)
    return abs(high - lower)


def get_asymmetry_factor(x: np.ndarray, y: np.ndarray, height: float | int = 0.1) -> float:
    """ Calculates asymmetry factor at height. """
    lower, high = get_width_at(x, y, height)
    middle = x[np.argmax(y)]

    return (high - middle) / (middle - lower)


def get_width_at(x: np.ndarray, y: np.ndarray, height: float | int = 0.5) -> tuple[float, float]:
    """ Determine full-width-x_max of a peaked set of points, x and y. """
    x, y = check_for_flip(x, y)
    height_half_max = np.max(y) * height
    index_max = np.argmax(y)
    if index_max == 0 or index_max == len(x):  # peak max is at end.
        logging.info("Finding fwhm is not possible with a peak max at an bound.")
        return 0, 0

    x_low = np.interp(height_half_max, y[:index_max], x[:index_max])
    x_high = np.interp(height_half_max, np.flip(y[index_max:]), np.flip(x[index_max:]))

    if x_low == x[0]:
        logging.info("fwhm or asym is having to linear interpolate on the lower end.")
        slice_ = max(3, int(index_max / 10))
        fit = np.polyfit(y[:slice_], x[:slice_], deg=1)
        p = np.poly1d(fit)
        x_low = p(height_half_max)

    if x_high == x[-1]:
        logging.info("fwhm or asym is having to linear interpolate on the lower end.")
        slice_ = max(3, int(index_max / 10))
        fit = np.polyfit(y[-slice_:], x[-slice_:], deg=1)
        p = np.poly1d(fit)
        x_high = p(height_half_max)

    return x_low, x_high
