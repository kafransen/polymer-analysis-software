from __future__ import annotations
import pathlib

import numpy as np

from chem_analysis.base_obj.signal_array import SignalArray
from chem_analysis.sec.sec_calibration import SECCalibration
from chem_analysis.sec.sec_signal import SECSignal, SECTypes
from chem_analysis.analysis.peak_SEC import PeakSEC


class SECSignalArray(SignalArray):
    TYPES_ = SECTypes
    _peak_type = PeakSEC

    def __init__(self,
                 x_raw: np.ndarray,
                 time_raw: np.ndarray,
                 data_raw: np.ndarray,
                 calibration: SECCalibration = None,
                 type_: SECTypes = SECTypes.UNKNOWN,
                 x_label: str = None,
                 y_label: str = None,
                 z_label: str = None,
                 name: str = None
                 ):
        x_label = x_label or "retention_time"
        y_label = y_label or "time"
        z_label = z_label or "signal"
        super().__init__(x_raw, time_raw, data_raw, x_label, y_label, z_label, name)
        self.calibration = calibration
        self.type_ = type_

    def get_signal(self, index: int, processed: bool = False) -> SECSignal:
        if processed:
            sig = SECSignal(x_raw=self.x, y_raw=self.data[index, :], calibration=self.calibration, type_=self.type_,
                            x_label=self.x_label, y_label=self.y_label, name=f"time: {self.time[index]}", id_=index)
        else:
            sig = SECSignal(x_raw=self.x_raw, y_raw=self.data_raw[index, :], calibration=self.calibration,
                            type_=self.type_,
                            x_label=self.x_label, y_label=self.y_label, name=f"time: {self.time[index]}", id_=index)
            sig.processor = self.processor.get_copy()

        sig.time = self.time[index]
        return sig

    @classmethod
    def from_file(cls, path: str | pathlib.Path, calibration: SECCalibration = None) -> SECSignalArray:
        class_ = super().from_file(path)
        class_.calibration = calibration

        return class_
