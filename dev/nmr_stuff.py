from typing import Iterable
from dataclasses import dataclass
import pathlib
from datetime import datetime
import math

import numpy as np

import plotly.graph_objs as go


@dataclass(slots=True)
class NMRParameters:
    is2D: bool = None
    sizeTD1: int = None
    spectral_width: float = None
    sizeTD2: int = None
    carrier: float = None
    date_start: datetime = None
    number_scans: int = None
    solvent: str = None
    temperature_coil: float = None  # Kelvin
    instrument: str = None
    pulse_sequence: str = None
    probe: str = None
    receiver_gain: float = None
    spectrometer_frequency: float = None
    spectral_size: int = None
    top_spin_version: str = None
    instrument_position: int = None
    shift_points: int = None
    nucleus: str = None

    @property
    def dwell_time(self) -> float:
        return 1 / self.spectral_width


parseDict = {3: "i4", 4: "f8"}


def get_line(lines: Iterable[str], startswith: str) -> str | None:
    for line in lines:
        if line.startswith(startswith):
            return line

    return None


def parse_acqus_file(path: pathlib.Path) -> NMRParameters:
    """

    Parameters
    ----------
    path:
        should finish with "/acqus"

    Returns
    -------

    """
    with open(path, mode='r') as f:
        text = f.read()

    lines = text.split("\n")
    acqusfile = NMRParameters()

    line = get_line(lines, "##$SW_h=").strip("##$SW_h= ")
    acqusfile.spectral_width = float(line)

    line = get_line(lines, "##$TD").strip("##$TD= ")
    acqusfile.sizeTD2 = int(int(line) / 2)

    line = get_line(lines, "##$TD").strip("##$TD= ")
    acqusfile.carrier = float(line) * 1e6

    line = get_line(lines, "##$DATE_START").strip("##$DATE_START= ")
    acqusfile.date_start = datetime.fromtimestamp(int(line))

    line = get_line(lines, "##$NS=").strip("##$NS= ")
    acqusfile.number_scans = int(line)

    line = get_line(lines, "##$SOLVENT=").strip("##$SOLVENT= ").replace("<", "").replace(">", "")
    acqusfile.solvent = line

    line = get_line(lines, "##$shimCoilTempK=").strip("##$shimCoilTempK= ")
    acqusfile.temperature_coil = float(line)

    line = get_line(lines, "##$INSTRUM=").strip("##$INSTRUM= ").replace("<", "").replace(">", "")
    acqusfile.instrument = line

    line = get_line(lines, "##$PULPROG=").strip("##$PULPROG= ").replace("<", "").replace(">", "")
    acqusfile.pulse_sequence = line

    line = get_line(lines, "##$PROBHD=").strip("##$PROBHD= ").replace("<", "").replace(">", "")
    acqusfile.probe = line

    line = get_line(lines, "##$RG=").strip("##$RG= ")
    acqusfile.receiver_gain = float(line)

    line = get_line(lines, "##$SFO1=").strip("##$SFO1= ")
    acqusfile.spectrometer_frequency = float(line)

    line = get_line(lines, "##$TD=").strip("##$TD= ")
    acqusfile.spectral_size = float(line)

    line = get_line(lines, "##TITLE").strip("##TITLE= Parameter file, TopSpin ")
    acqusfile.top_spin_version = line

    line = get_line(lines, "##$HOLDER").strip("##$HOLDER= ")
    acqusfile.instrument_position = int(line)

    line = get_line(lines, "##$GRPDLY").strip("##$GRPDLY= ")
    acqusfile.shift_points = int(math.floor(float(line)))

    line = get_line(lines, "##$NUC1").strip("##$NUC1= ").replace("<", "").replace(">", "")
    acqusfile.nucleus = line

    return acqusfile


# def process_acqus2_file():
#     acqu2sFile = open(directory + "/acqu2s", mode='r')
#     self.files.append(acqu2sFile)
#     acqu2File = open(directory + "/acqu2", mode='r')
#     self.files.append(acqu2File)
#
#     self.is2D = True

# self.sizeTD1 = int(line[1])
# if self.debug:
#     print("sizeTD1: ", self.sizeTD1)
# elif line[0] == "##$SW_h":
#     self.sweepWidthTD1 = int(float(line[1]))

class Processor:
    def __init__(self, operationStack):
        """Define a pyNMR Processor. A processor is a list of operations
        that can be applied to NMR data."""

        self.operationStack = operationStack

    def runStack(self, nmrData, endpoint=-1, startpoint=0):
        nmrData.reset()

        if endpoint > 0:
            opList = self.operationStack[:endpoint]
        else:
            opList = self.operationStack

        for op in opList:
            # print(op.name)
            op.run(nmrData)

    def __getitem__(self, index):
        return self.operationStack[index]


class NMR:
    def __init__(self, parameters: NMRParameters = None):
        self._fid = None
        self._spectrum = None
        self._FID_time_axis = None
        self._ppm_axis = None
        self.parameters = parameters
        self.processing = Processor()

    def __repr__(self):
        return f"{self.parameters.nucleus} (in {self.parameters.solvent})"

    def __str__(self):
        return self.__repr__()

    @property
    def FID(self) -> np.ndarray:
        return self._fid

    @property
    def FID_real(self) -> np.ndarray:
        """ y axis of FID for visualization """
        return np.real(self._fid)

    @property
    def FID_time_axis(self) -> np.ndarray:
        """ x axis of FID """
        if self._FID_time_axis is None:
            self._FID_time_axis = np.linspace(
                0,
                (self.parameters.sizeTD2 - 1) * self.parameters.dwell_time,
                self.parameters.sizeTD2
            )

        return self._FID_time_axis

    @property
    def spectrum(self) -> np.ndarray:
        return 1

    @property
    def ppm_axis(self) -> np.ndarray:
        if self._ppm_axis is None:
            self._ppm_axis = np.linspace(
                -self.parameters.spectral_width / 2,
                self.parameters.spectral_width / 2,
                self.parameters.sizeTD2
            )

        return self._ppm_axis

    def load_from_raw_FID_data(self, data: np.ndarray):
        _real = data[0:self.parameters.sizeTD2 * 2:2]
        _imag = np.multiply(data[1:self.parameters.sizeTD2 * 2 + 1:2], 1j)
        self._fid = np.add(_real, _imag)

    @classmethod
    def from_bruker(cls, path: pathlib.Path) -> NMR:
        # load data from file
        parameters = parse_acqus_file(path / "acqus")
        data = get_fid(path / "fid")

        # construct NMR object
        nmr = NMR(parameters=parameters)
        nmr.load_from_raw_FID_data(data)
        return nmr

    def default_processing(self):
        self.processing.operationStack = [
            OPS.LeftShift(self.parameters.shift_points),
            OPS.LineBroadening(0.0),
            OPS.FourierTransform(),
            OPS.Phase0D(-90),
            OPS.Phase1D(self.parameters.shift_points, unit="time")
        ]


def get_fid(path: pathlib.Path, endianess: str = "<", dtype: np.dtype = np.dtype("f8")) -> np.ndarray:
    dtype_ = dtype.newbyteorder(endianess)
    with open(path, mode='rb') as f:
        d = f.read()
        return np.frombuffer(d, dtype=dtype_)


def main():
    path = pathlib.Path(r"C:\Users\nicep\Desktop\New folder\DW2-3_1\DW2-3\1")
    nmr = NMR.from_bruker(path)

    fig = go.Figure(
        go.Scatter(x=nmr.FID_time_axis, y=nmr.FID_real)
    )
    fig.write_html("temp.html", auto_open=True)


if __name__ == "__main__":
    main()
