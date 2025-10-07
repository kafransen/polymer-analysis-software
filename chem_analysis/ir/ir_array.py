
import numpy as np

from chem_analysis.base_obj.signal_array import SignalArray
from chem_analysis.ir.ir_signal import IRSignal


class IRSignalArray(SignalArray):
    _signal = IRSignal

    def __init__(self,
                 x_raw: np.ndarray,
                 time_raw: np.ndarray,
                 data_raw: np.ndarray,
                 x_label: str = None,
                 y_label: str = None,
                 z_label: str = None,
                 name: str = None
                 ):
        x_label = x_label or "wave_number"
        y_label = y_label or "time"
        z_label = z_label or "absorbance"
        super().__init__(x_raw, time_raw, data_raw, x_label, y_label, z_label, name)

    @property
    def cm_1(self) -> np.ndarray:
        return self.x

    @property
    def micrometer(self) -> np.ndarray:
        return 1 / self.x * 1000

    @property
    def absorbance(self) -> np.ndarray:
        return self.data

    @property
    def transmittance(self) -> np.ndarray:
        return np.exp(-self.data)

    def get_signal(self, index: int, processed: bool = False) -> IRSignal:
        if processed:
            sig = IRSignal(x_raw=self.x, y_raw=self.data[index, :], x_label=self.x_label, y_label=self.y_label,
                       name=f"time: {self.time[index]}", id_=index)
        else:
            sig = IRSignal(x_raw=self.x_raw, y_raw=self.data_raw[index, :], x_label=self.x_label, y_label=self.y_label,
                       name=f"time: {self.time[index]}", id_=index)
            sig.processor = self.processor.get_copy()

        sig.time = self.time[index]
        return sig
