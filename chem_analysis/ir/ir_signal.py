import numpy as np

from chem_analysis.base_obj.signal_ import Signal


class IRSignal(Signal):
    """ IR Signal
    """
    def __init__(self,
                 x_raw: np.ndarray,
                 y_raw: np.ndarray,
                 x_label: str = None,
                 y_label: str = None,
                 name: str = None,
                 id_: int = None
                 ):
        x_label = x_label or "wave_number"
        y_label = y_label or "absorbance"
        super().__init__(x_raw, y_raw, x_label, y_label, name, id_)

    @property
    def cm_1(self) -> np.ndarray:
        return self.x

    @property
    def micrometer(self) -> np.ndarray:
        return 1 / self.x * 1000

    @property
    def absorbance(self) -> np.ndarray:
        return self.y

    @property
    def transmittance(self) -> np.ndarray:
        return np.exp(-self.y)
