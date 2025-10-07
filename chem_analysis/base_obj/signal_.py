from typing import Sequence
import pathlib

import numpy as np

import chem_analysis.utils.math as general_math
from chem_analysis.processing.base import Processor
from chem_analysis.analysis.peak import PeakBounded


class Signal:
    """ signal

    A signal is any x-y data.

    Attributes
    ----------
    name: str
        Any name the user wants to add.
    x_label: str
        x-axis label
    y_label: str
        y-axis label
    """
    __count = 0
    _peak_type = PeakBounded

    def __init__(self,
                 x_raw: np.ndarray,
                 y_raw: np.ndarray,
                 x_label: str = None,
                 y_label: str = None,
                 name: str = None,
                 id_: int = None
                 ):
        """

        Parameters
        ----------
        x_raw: np.ndarray
            raw x data
        y_raw: np.ndarray
            raw y data
        x_label: str
            x-axis label
        y_label: str
            y-axis label
        name: str
            user defined name

        Notes
        -----
        * Either 'ser' or 'x' and 'y' are required but not both.

        """
        # flip data if giving backwards; it should be low to high
        if x_raw[1] > x_raw[-1]:
            x_raw = np.flip(x_raw)
            y_raw = np.flip(y_raw)
        if len(x_raw.shape) != 1:
            raise ValueError(f"'x_raw' must shape 1. \n\treceived: {x_raw.shape}")
        if len(y_raw.shape) != 1:
            raise ValueError(f"'y_raw' must shape 1. \n\treceived: {y_raw.shape}")
        if x_raw.shape != y_raw.shape:
            raise ValueError(f"'x_raw' and 'y_raw' must have same shape. \n\treceived: x_raw:{x_raw.shape} || y_raw: "
                             f"{y_raw.shape}")

        self.x_raw = x_raw
        self.y_raw = y_raw
        self.id_ = id_ if id_ is not None else Signal.__count
        Signal.__count += 1
        self.name = name if name is not None else f"signal_{self.id_}"
        self.x_label = x_label if x_label is not None else "x_axis"
        self.y_label = y_label if y_label is not None else "y_axis"

        self.processor = Processor()
        self._x = None
        self._y = None

    def __repr__(self):
        text = f"{self.name}: "
        text += f"{self.x_label} vs {self.y_label}"
        text += f" (pts: {len(self)})"
        return text

    def __len__(self) -> int:
        return len(self.x)

    @property
    def x(self) -> np.ndarray:
        if not self.processor.processed:
            self._x, self._y = self.processor.run(self.x_raw, self.y_raw)

        return self._x

    @property
    def y(self) -> np.ndarray:
        if not self.processor.processed:
            self._x, self._y = self.processor.run(self.x_raw, self.y_raw)

        return self._y

    def y_normalized_by_max(self, x_range: Sequence[int | float] = None) -> np.ndarray:
        if x_range is None:
            return self.y/np.max(self.y)

        slice_ = general_math.get_slice(self.x, *x_range)
        return general_math.normalize_by_max(self.y[slice_])

    def y_normalized_by_area(self, x_range: Sequence[int | float] = None) -> np.ndarray:
        if x_range is None:
            return general_math.normalize_by_area(self.x, self.y)

        slice_ = general_math.get_slice(self.x, *x_range)
        return general_math.normalize_by_area(self.x[slice_], self.y[slice_])

    @classmethod
    def from_file(cls, path: str | pathlib.Path):
        if isinstance(path, str):
            path = pathlib.Path(path)

        if path.suffix == ".csv":
            x, y, x_label, y_label = load_csv(path)
        elif path.suffix == ".feather":
            from chem_analysis.utils.feather_format import feather_to_numpy
            data, headers = feather_to_numpy(path)
            x, y = data[:, 0], data[:, 1]
            if headers[0] != "0":
                x_label = headers[0]
                y_label = headers[1]
            else:
                x_label = y_label = None
        elif path.suffix == ".npy":
            x, y = np.load(str(path))
            x_label = y_label = None
        else:
            raise NotImplementedError("File type currently not supported.")

        return cls(x, y, x_label=x_label, y_label=y_label)

    def to_feather(self, path: str | pathlib.Path):
        from chem_analysis.utils.feather_format import numpy_to_feather

        headers = [self.x_label, self.y_label]
        numpy_to_feather(np.column_stack((self.x, self.y)), path, headers=headers)

    def to_csv(self, path: str | pathlib.Path, headers: bool = False, encoding: str = "utf-8"):
        kwargs = {"encoding": encoding}
        if headers:
            kwargs["headers"] = [self.x_label, self.y_label]
        np.savetxt(path, np.column_stack((self.x, self.y)), delimiter=",", **kwargs)

    def to_npy(self, path: str | pathlib.Path, **kwargs):
        np.save(path, np.column_stack((self.x, self.y)), **kwargs)


def load_csv(path: pathlib):
    import csv

    # Initialize variables to store data
    data = []
    x_label = None
    y_label = None

    # Read CSV file
    with open(path, 'r') as file:
        csv_reader = csv.reader(file)

        # Check if the first row contains numbers
        first_row = next(csv_reader, None)

        if len(first_row) != 2:
            raise ValueError("Data not correct format.")

        if any(cell.isalpha() for cell in first_row):
            # If the first row contains non-numeric values, consider it as column labels
            x_label, y_label = first_row
        else:
            # If the first row contains numbers, treat them as data and set labels to None
            data.append([float(cell) for cell in first_row])

        # Read the remaining rows
        for row in csv_reader:
            data.append([float(cell) for cell in row])

    # Convert data to NumPy array
    data_array = np.array(data)

    return data_array[:, 0], data_array[:, 1], x_label, y_label
