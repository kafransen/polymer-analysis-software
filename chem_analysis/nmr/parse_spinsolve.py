from dataclasses import dataclass
import pathlib
from datetime import datetime, timedelta

import numpy as np

from chem_analysis.nmr.parameters import NMRParameters, NMRExperiments


@dataclass(slots=True)
class SpinSolveParameters:
    Solvent: str
    Sample: str
    Custom: str
    startTime: datetime
    PreProtocolDelay: int
    acqDelay: float
    b1Freq: float
    bandwidth: float
    dwellTime: timedelta
    experiment: str
    expName: str
    nrPnts: int
    nrScans: int
    repTime: timedelta
    rxChannel: str
    rxGain: int
    lowestFrequency: float
    totalAcquisitionTime: float
    graphTitle: str
    linearPrediction: str
    userData: str
    s_90Amplitude: int  # variables can't start with numbers so add prefix
    pulseLength: float
    pulseAngle: int
    ComputerName: str
    UserName: str
    SpinsolveUser: str
    ProtocolDataID: int
    Protocol: str
    Options: str
    Spectrometer: str
    InstrumentType: str
    InstrumentCode: str
    Software: str
    WindowsLoggedUser: str
    BackupLocation: str
    Shim_Timestamp: datetime
    Shim_Width50: float
    Shim_Width055: float
    Shim_SNR: float
    Shim_ReferencePeakIndex: float
    Shim_ReferencePPM: float
    Reference_Timestamp: datetime
    Reference_File: str
    StartProtocolTemperatureMagnet: float
    StartProtocolTemperatureBox: float
    StartProtocolTemperatureRoom: float
    CurrentTemperatureMagnet: float
    CurrentTemperatureBox: float
    CurrentTemperatureRoom: float
    CurrentTime: datetime


parsing_functions = {
    "startTime": lambda x: datetime.fromisoformat(x),
    "dwellTime": lambda x: timedelta(milliseconds=x),
    "repTime": lambda x: timedelta(milliseconds=x),
    "pulseLength": lambda x: timedelta(milliseconds=x),
    "CurrentTemperatureMagnet": lambda x: x + 273.15,
    "Shim_Timestamp": lambda x: datetime.fromisoformat(x),
    "Reference_Timestamp": lambda x: datetime.fromisoformat(x),
    "CurrentTime": lambda x: datetime.fromisoformat(x)
}


def parse_acqu_file(path: pathlib.Path) -> SpinSolveParameters:
    """

    Parameters
    ----------
    path:
        should finish with "/acqu.par"

    Returns
    -------

    """
    with open(path / "acqu.par", mode='r') as f:
        text = f.read()

    lines = text.strip().split("\n")
    parameters = dict()

    for line in lines:
        name, value = line.split("=")
        name = name.strip()
        if name[0].isdigit():
            name = "s_" + name  # variables can't start with numbers so add prefix
        value = value.strip()

        # convert to types
        if '"' in value:
            value = value.replace('"', "")
        elif value[0].isdigit():
            value = float(value)
            if value == int(value):
                value = int(value)
        if name in parsing_functions:
            value = parsing_functions[name](value)

        parameters[name] = value

    return SpinSolveParameters(**parameters)


def get_nmr_experiment(parameters: SpinSolveParameters) -> NMRExperiments:
    if parameters.Protocol == "1D EXTENDED+":
        return NMRExperiments.PROTON

    return NMRExperiments.UNKNOWN


def parse_spinsolve_parameters(path: pathlib.Path) -> NMRParameters:
    spinsolve = parse_acqu_file(path)
    parameters = NMRParameters()

    parameters.solvent = spinsolve.Solvent
    parameters.sample_name = spinsolve.Sample
    parameters.date_start = spinsolve.startTime
    parameters.date_end = spinsolve.CurrentTime
    parameters.spectrometer_frequency = spinsolve.b1Freq
    parameters.type_ = get_nmr_experiment(spinsolve)

    parameters.number_scans = spinsolve.nrScans
    parameters.pulse_width = spinsolve.pulseLength  # TODO: check
    parameters.repetition_delay = spinsolve.dwellTime
    parameters.acquisition_time = spinsolve.repTime - spinsolve.dwellTime
    parameters.pulse_angle = spinsolve.pulseAngle
    parameters.number_points = spinsolve.nrPnts

    return parameters


def get_spinsolve_data(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray, bool]:
    if (path / "nmr_fid.dx").exists():
        pass  # TODO: JCAMP-DX is the IUPAC standard format https://iupac.org/what-we-do/digital-standards/jcamp-dx/
        # return x, y, True

    # 1d files
    options = ["data.1d", "spectrum.1d", "spectrum_processed.1d"]
    for option in options:
        path_ = path / option
        if path_.exists():
            break
    else:
        raise ValueError("No valid data file found")

    with open(path_, "rb") as f:
        raw_data = f.read()

    # first 32 bytes are parameters
    keys = ["owner", "format", "version", "dataType", "xDim", "yDim", "zDim", "qDim"]
    dict_ = dict()
    for i, k in enumerate(keys):
        dict_[k] = int.from_bytes(raw_data[i * 4:i * 4 + 4], "little")
    data = np.frombuffer(raw_data[32:], "<f")

    # The first 1/3 of the file is xaxis
    split = int(data.shape[-1] / 3)
    x = data[0: split]

    # Then real and imaginary data points interleaved
    y = data[split:: 2] + 1j * data[split + 1:: 2]

    return x, y, False


def get_spinsolve_data_csv(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path / "spectrum_processed.csv", delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1]


def is_spin_solve_file(path: pathlib.Path) -> bool:
    target_string = "Spinsolve"
    try:
        with open(path / "nmr_fid.dx", 'r') as file:
            for line_number, line in enumerate(file):
                if target_string in line:
                    return True
                if line_number > 10:
                    break
    except FileNotFoundError:
        pass

    return False
