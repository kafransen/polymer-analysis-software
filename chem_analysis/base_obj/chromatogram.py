from collections import OrderedDict

import numpy as np

from chem_analysis.base_obj.signal_ import Signal
from chem_analysis.utils.printing_tables import StatsTable


class Chromatogram:
    """
    A grouping of Signals that occur over the same time interval.
    """
    __count = 0

    def __init__(self, data: list[Signal] | np.ndarray, name: str = None):
        if isinstance(data, list):
            if all(isinstance(dat, Signal) for dat in data):
                data = {dat.y_label if dat.y_label is not None else f"y_axis{i}": dat for i, dat in enumerate(data)}
            else:
                raise ValueError("Invalid type in list")

        self.signals = data
        for sig in self.signals:
            setattr(self, sig.name, sig)

        if name is None:
            name = f"Chromat_{Chromatogram.__count}"
            Chromatogram.__count += 1
        self.name = name

    def __repr__(self) -> str:
        text = f"{self.name}: "
        text += "; ".join(self.names)
        return text

    @property
    def names(self):
        return [i.name for i in self.signals]

    @property
    def y_labels(self):
        return [i.y_label for i in self.signals]

    @property
    def x_label(self):
        return self.signals[0].x_label

    @property
    def number_of_signals(self):
        return len(self.signals)

    def get_stats(self) -> list[OrderedDict]:
        dicts_ = []
        for sig in self.signals:
            dicts_.append(sig.get_stats())

        return dicts_

    def stats_table(self) -> StatsTable:
        return StatsTable.from_list_dicts(self.get_stats())
