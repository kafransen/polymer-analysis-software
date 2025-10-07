from typing import Iterable
from dataclasses import dataclass
import pathlib
from datetime import timedelta, datetime
import math

import numpy as np

from chem_analysis.nmr.parameters import NMRParameters


@dataclass(slots=True)
class NMRParametersBruker(NMRParameters):
    sizeTD1: int = None
    spectral_width: float = None
    sizeTD2: int = None
    carrier: float = None
    temperature_coil: float = None  # Kelvin
    instrument: str = None
    pulse_sequence: str = None
    probe: str = None
    receiver_gain: float = None
    spectrometer_frequency: float = None
    # spectral_size: int = None
    top_spin_version: str = None
    instrument_position: int = None
    shift_points: int = None
    nucleus: str = None


parseDict = {3: "i4", 4: "f8"}


def parse_bruker_folder(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray, NMRParametersBruker]:
    parameters = parse_acqus_file(path / "acqus")
    y = get_fid(path / "fid")
    x = parameters.compute_time() #np.linspace(0, (parameters.sizeTD2-1)*parameters.dwellTime, num=parameters.sizeTD2)
    return x, y, parameters


def get_fid(path: pathlib.Path, endianess: str = "<", dtype: np.dtype = np.dtype("f8")) -> np.ndarray:
    dtype_ = dtype.newbyteorder(endianess)
    with open(path, mode='rb') as f:
        d = f.read()
        return np.frombuffer(d, dtype=dtype_)


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
    parameters = NMRParametersBruker()


    # TD1 is number of FIDs, TD2 is number of datapoints in each FID
    line = get_line(lines, "##$TD").strip("##$TD= ")
    parameters.number_points = int(int(line)/2)

    line = get_line(lines, "##$SW_h=").strip("##$SW_h= ")
    parameters.acquisition_time = timedelta(seconds=parameters.number_points/float(line))

    line = get_line(lines, "##$DATE").strip("##$DATE= ")
    parameters.date_start = datetime.fromtimestamp(int(line))

    line = get_line(lines, "##$NS=").strip("##$NS= ")
    parameters.number_scans = int(line)

    line = get_line(lines, "##$SOLVENT=").strip("##$SOLVENT= ").replace("<", "").replace(">", "")
    parameters.solvent = line

    # line = get_line(lines, "##$shimCoilTempK=").strip("##$shimCoilTempK= ")
    # parameters.temperature_coil = float(line)

    line = get_line(lines, "##$INSTRUM=").strip("##$INSTRUM= ").replace("<", "").replace(">", "")
    parameters.instrument = line

    line = get_line(lines, "##$PULPROG=").strip("##$PULPROG= ").replace("<", "").replace(">", "")
    parameters.pulse_sequence = line

    line = get_line(lines, "##$PROBHD=").strip("##$PROBHD= ").replace("<", "").replace(">", "")
    parameters.probe = line

    line = get_line(lines, "##$RG=").strip("##$RG= ")
    parameters.receiver_gain = float(line)

    line = get_line(lines, "##$SFO1=").strip("##$SFO1= ")
    parameters.spectrometer_frequency = float(line)


    line = get_line(lines, "##TITLE").strip("##TITLE= Parameter file, TopSpin ")
    parameters.top_spin_version = line

    line = get_line(lines, "##$HOLDER").strip("##$HOLDER= ")
    parameters.instrument_position = int(line)

    line = get_line(lines, "##$GRPDLY").strip("##$GRPDLY= ")
    parameters.shift_points = int(math.floor(float(line)))

    line = get_line(lines, "##$NUC1").strip("##$NUC1= ").replace("<", "").replace(">", "")
    parameters.nucleus = line

    return parameters
