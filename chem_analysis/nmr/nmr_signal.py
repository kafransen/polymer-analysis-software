from __future__ import annotations
import pathlib

import numpy as np

from chem_analysis.nmr.parameters import NMRParameters
from chem_analysis.base_obj.signal_ import Signal


class NMRFID(Signal):
    """
    Free Induction Decay (FID)
    """

    def __init__(self,
                 x: np.ndarray,
                 y: np.ndarray,
                 parameters: NMRParameters = None,
                 x_label: str = None,
                 y_label: str = None,
                 name: str = None,
                 id_: int = None
                 ):
        x_label = x_label or "time"
        y_label = y_label or "signal"
        super().__init__(x, y, x_label, y_label, name, id_)
        self.parameters = parameters

    @property
    def FID_real(self) -> np.ndarray:
        """ y axis of FID for visualization """
        return np.real(self.y)

    # def default_processing(self):
    #     from chem_analysis.processing.fourier_transform import fft
    #     self.processor.add(LeftShift(self.parameters.shift_points))
    #     self.processor.add(LineBroadening(0.0))
    #     self.processor.add(FourierTransform())

    def generate_nmr(self) -> NMRSignal:
        self.default_processing()
        return NMRSignal(x_raw=self.x, y_raw=self.y, parameters=self.parameters)


def load_from_raw_FID_data(data: np.ndarray, parameters: NMRParameters):
    _real = data[0:parameters.number_points * 2:2]
    _imag = np.multiply(data[1:parameters.number_points * 2 + 1:2], 1j)
    return np.add(_real, _imag)


class NMRSignal(Signal):
    def __init__(self,
                 x_raw: np.ndarray,
                 y_raw: np.ndarray,
                 parameters: NMRParameters = None,
                 x_label: str = None,
                 y_label: str = None,
                 name: str = None,
                 id_: int = None
                 ):
        x_label = x_label or "ppm"
        y_label = y_label or "signal"
        super().__init__(x_raw, y_raw, x_label, y_label, name, id_)

        self.fid: NMRFID | None = None
        self.parameters = parameters

    def __repr__(self):
        return f"{self.parameters.type_.name} (in {self.parameters.solvent})"

    def __str__(self):
        return self.__repr__()

    def _fid_processing(self):
        if self.fid is None:
            return

        if not self.fid.processor.processed:
            self.x_raw, self.y_raw = self.fid.x, self.fid.y

    @property
    def x(self) -> np.ndarray:
        self._fid_processing()
        if not self.processor.processed:
            self._x, self._y = self.processor.run(self.x_raw, self.y_raw)

        return self._x

    @property
    def y(self) -> np.ndarray:
        self._fid_processing()
        if not self.processor.processed:
            self._x, self._y = self.processor.run(self.x_raw, self.y_raw)

        return self._y

    # def default_processing(self):
    #     from chem_analysis.processing import
    #     self.processor.add(Phase0D(-90))
    #     self.processor.add(Phase1D(self.parameters.shift_points, unit="time"))

    @classmethod
    def from_bruker(cls, path: pathlib.Path) -> NMRSignal:
        if isinstance(path, str):
            path = pathlib.Path(path)

        from chem_analysis.nmr.parse_bruker import parse_bruker_folder
        x, y, parameters = parse_bruker_folder(path)

        fid = NMRFID(x, y, parameters=parameters)
        return fid.generate_nmr()

    @classmethod
    def from_spinsolve(cls, path: pathlib.Path | str) -> NMRSignal:
        if isinstance(path, str):
            path = pathlib.Path(path)

        from chem_analysis.nmr.parse_spinsolve import parse_spinsolve_parameters, get_spinsolve_data

        # load data from file
        parameters = parse_spinsolve_parameters(path)
        x, y, is_fid = get_spinsolve_data(path)

        if is_fid:
            fid = NMRFID(x, y, parameters=parameters)
            return fid.generate_nmr()

        return NMRSignal(x_raw=x, y_raw=y, parameters=parameters)

    @classmethod
    def from_spinsolve_csv(cls, path: pathlib.Path | str) -> NMRSignal:
        if isinstance(path, str):
            path = pathlib.Path(path)

        from chem_analysis.nmr.parse_spinsolve import parse_spinsolve_parameters, get_spinsolve_data_csv

        # load data from file
        parameters = parse_spinsolve_parameters(path)
        x, y = get_spinsolve_data_csv(path)

        return NMRSignal(x_raw=x, y_raw=y, parameters=parameters)
