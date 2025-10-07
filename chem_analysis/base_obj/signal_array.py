import pathlib
from typing import Sequence, Iterable

import numpy as np

from chem_analysis.processing.base import Processor
from chem_analysis.base_obj.signal_ import Signal


class SignalArray:
    """
    A grouping of Signals where each Signal occurred at a different time interval.
    """
    _signal = Signal
    _peak_type = _signal._peak_type

    def __init__(self,
                 x_raw: np.ndarray,
                 time_raw: np.ndarray,
                 data_raw: np.ndarray,
                 x_label: str = None,
                 y_label: str = None,
                 z_label: str = None,
                 name: str = None
                 ):
        self.name = name
        self.x_raw = x_raw
        self.time_raw = time_raw
        self.data_raw = data_raw
        self.x_label = x_label or "x_axis"
        self.y_label = y_label or "time"
        self.z_label = z_label or "z_axis"

        self.processor = Processor()
        self._x = None
        self._time = None
        self._data = None

    def _process(self):
        self._x, self._time, self._data = self.processor.run(self.x_raw, self.time_raw, self.data_raw)

    @property
    def x(self) -> np.ndarray:
        if not self.processor.processed:
            self._process()

        return self._x

    @property
    def time(self) -> np.ndarray:
        if not self.processor.processed:
            self._process()

        return self._time

    @property
    def time_zeroed(self) -> np.ndarray:
        if not self.processor.processed:
            self._process()

        return self._time - self.time_raw[0]

    @property
    def data(self) -> np.ndarray:
        if not self.processor.processed:
            self._process()

        return self._data

    @property
    def number_of_signals(self):
        return len(self.time_raw)

    def pop(self, index: int) -> Signal:
        sig = self.get_signal(index)
        self.delete(index)
        return sig

    def delete(self, index: int | Iterable):
        if isinstance(index, int):
            index = [index]
        index.sort(reverse=True)  # delete largest to smallest to avoid issue of changing index
        for i in index:
            self.data_raw = np.delete(self.data_raw, i, axis=0)
            self.time_raw = np.delete(self.time_raw, i)
            self.processor.processed = False

    def get_signal(self, index: int, processed: bool = False) -> Signal:
        if processed:
            sig = Signal(x_raw=self.x, y_raw=self.data[index, :], x_label=self.x_label,
                     y_label=self.y_label, name=f"time: {self.time[index]}", id_=index)
        else:
            sig = Signal(x_raw=self.x_raw, y_raw=self.data_raw[index, :], x_label=self.x_label,
                         y_label=self.y_label, name=f"time: {self.time[index]}", id_=index)
            sig.processor = self.processor.get_copy()
        sig.time_ = self.time[index]
        return sig

    @classmethod
    def from_signals(cls, signals: Sequence[Signal]):
        # TODO: add interplation option
        x = signals[0].x
        x_label = signals[0].x_label
        z_label = signals[0].y_label

        time_ = np.empty(len(signals))
        data = np.empty((len(signals), len(x)), dtype=signals[0].y.dtype)
        for i, sig in enumerate(signals):
            if np.all(sig.x != x):
                raise ValueError(f"Signal {i} has a different x-axis than first signal.")
            data[i, :] = sig.y
            if hasattr(sig, "time_"):
                time_[i] = sig.time_
            else:
                time_[i] = i

        return cls(x_raw=x, time_raw=time_, data_raw=data, x_label=x_label, z_label=z_label)

    @classmethod
    def from_file(cls, path: str | pathlib.Path):
        from chem_analysis.utils.feather_format import feather_to_numpy
        from chem_analysis.utils.math import unpack_time_series

        if isinstance(path, str):
            path = pathlib.Path(path)

        if path.suffix == ".csv":
            data = np.loadtxt(path, delimiter=",")
            x, time_, data = unpack_time_series(data)
            x_label = y_label = z_label = None

        elif path.suffix == ".feather":
            data, names = feather_to_numpy(path)
            x, time_, data = unpack_time_series(data)
            if names[0] != "0":
                x_label = names[0]
                y_label = names[1]
                z_label = names[2]
            else:
                x_label = y_label = z_label = None

        elif path.suffix == ".npy":
            data = np.load(str(path))
            x, time_, data = unpack_time_series(data)
            x_label = y_label = z_label = None
        else:
            raise NotImplemented("File type currently not supported.")

        return cls(x_raw=x, time_raw=time_, data_raw=data, x_label=x_label, y_label=y_label, z_label=z_label)

    def to_feather(self, path: str | pathlib.Path):
        from chem_analysis.utils.feather_format import numpy_to_feather
        from chem_analysis.utils.math import pack_time_series

        headers = list(str(0) for i in range(len(self.time)+1))
        headers[0] = self.x_label
        headers[1] = self.y_label
        headers[2] = self.z_label

        numpy_to_feather(pack_time_series(self.x, self.time, self.data), path, headers=headers)

    def to_csv(self, path: str | pathlib.Path, **kwargs):
        from chem_analysis.utils.math import pack_time_series

        if "encodings" not in kwargs:
            kwargs["encoding"] = "utf-8"
        if "delimiter" not in kwargs:
            kwargs["delimiter"] = ","

        np.savetxt(path, pack_time_series(self.x, self.time, self.data), **kwargs)  # noqa

    def to_npy(self, path: str | pathlib.Path, **kwargs):
        from chem_analysis.utils.math import pack_time_series

        np.save(path, pack_time_series(self.x, self.time, self.data), **kwargs)
