import abc

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter

from chem_analysis.processing.base import ProcessingMethod


class Smoothing(ProcessingMethod, abc.ABC):
    ...


class Gaussian(Smoothing):
    def __init__(self, sigma: float | int = 10):
        """

        Parameters
        ----------
        sigma
            Standard deviation for Gaussian kernel.
        """
        self.sigma = sigma

    def run(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return x, gaussian_filter(y, self.sigma)

    def run_array(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        for i in range(z.shape[0]):
            _, z[i, :] = self.run(x, z[i, :])
        return x, y, z

    # def run_2D(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    #     return x, y, gaussian_filter(z, self.sigma)


class SavitzkyGolay(Smoothing):
    """

    """
    def __init__(self, window_length: int = 10, order: int = 3):
        """
        The Savitzky Golay filter is a particular type of low-pass filter, well adapted for data smoothing.
        The Savitzky-Golay filter removes high frequency noise from data.
        It has the advantage of preserving the original shape and
        features of the signal better than other types of filtering
        approaches, such as moving averages techniques.

        Parameters
        ----------
        window_length:
            The length of the filter window (i.e., the number of coefficients)
            window_length must be less than or equal to the size of y
        order:
            The order of the polynomial used to fit the samples.
            order must be less than window_length.
        """
        if order > window_length:
            raise ValueError(f"'SavitzkyGolay.order'({order}) must be less than window_length ({window_length}).")
        self.window_length = window_length
        self.order = order

    def run(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.window_length > len(y):
            raise ValueError(f"'SavitzkyGolay.window_length'({self.window_length}) must be less than or "
                             f"equal to the size of y ({len(y)}).")
        return x, savgol_filter(y, self.window_length, self.order)

    def run_array(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.window_length > len(z.shape[0]):
            raise ValueError(f"'SavitzkyGolay.window_length'({self.window_length}) must be less than or "
                             f"equal to the first dimension of z ({len(z.shape[0])}).")
        return x, y, savgol_filter(z, self.window_length, self.order, axis=0)

    # def run_2D(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    #     # TODO: double check if this truely does 2D
    #     if self.window_length > len(z.shape[0]) or self.window_length > len(z.shape[1]):
    #         raise ValueError(f"'SavitzkyGolay.window_length'({self.window_length}) must be less than or "
    #                          f"equal to both dimensions of z ({len(z.shape)}).")
    #     return x, y, savgol_filter(z, self.window_length, self.order)


class ExponentialTime(Smoothing):
    def __init__(self, a: int | float = 0.8):
        """
        The exponential filter is a weighted combination of the previous estimate (output) with the newest input data,
        with the sum of the weights equal to 1 so that the output matches the input at steady state.

        Parameters
        ----------
        a:
            smoothing constant
            a is a constant between 0 and 1, normally between 0.8 and 0.99
        """
        self.a = a
        self._other_a = 1-a

    def run(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("Only valid for SignalArrays")

    def run_array(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        for row in range(1, z.shape[0]):
            z[row, :] = self.a * z[row - 1, :] + self._other_a * z[row, :]
        return x, y, z


class GaussianTime(Smoothing):
    def __init__(self, sigma: float | int = 10):
        """

        Parameters
        ----------
        sigma
            Standard deviation for Gaussian kernel.
        """
        self.sigma = sigma

    def run(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("Only valid for SignalArrays")

    def run_array(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        for row in range(z.shape[1]):
            z[:, row] = gaussian_filter(z[:, row], self.sigma)
        return x, y, z

    # def run_2D(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    #     return x, y, gaussian_filter(z, self.sigma)


class LineBroadening(Smoothing):
    def __init__(self, degree: float = 1):
        self.degree = degree  # Hz
        self._y_baseline = None

    @property
    def y_baseline(self) -> np.ndarray:
        return self._y_baseline

    def run(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        length = len(nmrData.allFid[-1][0])
        sp.multiply(nmrData.allFid[-1][:], sp.exp(-nmrData.fidTimeForLB[:length] * self.degree * np.pi))
        return x, y

# Savitzky-Golay
# moving average: span
# Whittaker Smoother: smooth factor 36
# Wavelets: scales=4, fraction=1%
# Cadzow

'''

import numpy as np


class Filter:

    def __init__(self):
        pass

    def process_data(self, data):
        pass


def savitzky_golay_filter_matrix(data: np.ndarray, window_size, order, axis=0, deriv=0, rate=1):
    """
    Applies the Savitzky-Golay filter to a matrix along a specified axis.

    Parameters
    ---------
    data:
        Input matrix of data points.
    window_size:
        Size of the smoothing window. It must be a positive odd number.
    order:
        Order of the polynomial to fit. It must be a non-negative integer.
    axis:
        The axis along which to apply the filter (default: 0).
    deriv:
        The order of the derivative to compute (default: 0).
    rate:
        The sampling rate of the input data (default: 1).

    Returns
    -------
    results:
        The filtered matrix.
    """
    try:
        window_size = np.abs(int(window_size))
        order = np.abs(int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")

    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")

    filtered_matrix = np.apply_along_axis(
        lambda x: savitzky_golay_filter(x, window_size, order, deriv=deriv, rate=rate),
                                          axis=axis, arr=data
    )

    return filtered_matrix


def savitzky_golay_filter(y, window_size, order, deriv=0, rate=1):
    """
    Applies the Savitzky-Golay filter to a one-dimensional array.

    Args:
        y (ndarray): Input one-dimensional array of data points.
        window_size (int): Size of the smoothing window. It must be a positive odd number.
        order (int): Order of the polynomial to fit. It must be a non-negative integer.
        deriv (int, optional): The order of the derivative to compute (default: 0).
        rate (int, optional): The sampling rate of the input data (default: 1).

    Returns:
        ndarray: The filtered array.

    Raises:
        ValueError: If `window_size` or `order` is not of type int.
        TypeError: If `window_size` is not a positive odd number or if it is too small for the given order.

    """
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * np.factorial(deriv)
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')





filter_exponential.min_number_points = 2


def filter_low_pass_fft(data: np.ndarray, cutoff_freq: int = 50, sampling_freq: int = 1000):
    """

    Parameters
    ----------
    data
    cutoff_freq
    sampling_freq

    Returns
    -------

    """
    # Calculate the normalized cutoff frequency
    normalized_cutoff_freq = cutoff_freq / (0.5 * sampling_freq)

    # Perform the Fourier transform
    fft = np.fft.fft(data)

    # Create the frequency axis
    freq = np.fft.fftfreq(len(data))

    # Apply the filter
    fft_filtered = np.where(np.abs(freq) <= normalized_cutoff_freq, fft, 0)

    # Perform the inverse Fourier transform
    filtered_signal = np.fft.ifft(fft_filtered)

    # Return the filtered signal
    return np.real(filtered_signal)

'''