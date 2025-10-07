import abc

import numpy as np

from chem_analysis.processing.base import ProcessingMethod


class Despike(ProcessingMethod, abc.ABC):

    @abc.abstractmethod
    def run(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ...


class RollingWindow(Despike):
    def __init__(self, window: int = 20, m: float = 2):
        self.window = window
        self.m = m

    @staticmethod
    def detect_outliers(data: np.ndarray, m: float):
        dist_from_median = np.abs(data - np.median(data))
        median_deviation = np.median(dist_from_median)
        if median_deviation != 0:
            scale_distances_from_median = dist_from_median / median_deviation
            return scale_distances_from_median > m  # True is an outlier

        return np.zeros_like(data)  # no outliers

    @staticmethod
    def window_calc(data: np.ndarray, pos: int, m: float) -> float:
        outlier = RollingWindow.detect_outliers(data, m)
        if outlier[pos]:
            return float(np.median(data[np.invert(outlier)]))
        else:
            return data[pos]

    def run(self, x: np.ndarray, y: np.ndarray, ) -> tuple[np.ndarray, np.ndarray]:
        out = np.empty_like(y)

        if self.window % 2 != 0:
            self.window += 1

        span = int(self.window / 2)
        for i in range(len(y)):
            if i < span:  # left edge
                out[i] = self.window_calc(y[:self.window], i, self.m)
            elif i > len(y) - span:  # right edge
                out[i] = self.window_calc(y[self.window:], i - (len(y) - self.window), self.m)
            else:  # middle
                out[i] = self.window_calc(y[i - span:i + span], span, self.m)

        return x, out

    @staticmethod
    def _test():
        import plotly.graph_objs as go

        n = 1000
        x = np.linspace(0, n - 1, n)
        y = np.ones(n) + np.random.random(n)
        y[100] = 5
        y[101] = 3

        y[500] = 2
        y[501] = 1.7
        y[502] = 1.5
        y[503] = 1.2

        y[700] = 0.5
        y[701] = 0
        y[702] = -0.5
        y[703] = -1

        algorithm = RollingWindow()
        x, y_new = algorithm.run(x, y)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines"))
        fig.add_trace(go.Scatter(x=x, y=y_new, mode="lines"))
        fig.write_html("temp.html", auto_open=True)


# Denoise
# non-local means: noise factor 0.75 blockwise
# median-modified wiener
# gaussian