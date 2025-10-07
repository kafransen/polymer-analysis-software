
import os
import pathlib
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pyarrow as pa


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


def process_one(path: pathlib.Path) -> tuple[float, np.ndarray]:
    parameters = parse_acqu_file(path)

    data = np.loadtxt(path / "spectrum_processed.csv", delimiter=",", skiprows=1)

    return parameters.CurrentTime.timestamp(), data


def process_many(path: pathlib.Path):
    files = tuple(os.scandir(path))
    files = sorted(files, key=lambda x: int(x.name[1:]))
    times = np.empty(len(files))
    ppm = None
    data = None
    ppm_slice = slice(-2, 12)  # ppm
    ppm_slice_index = None
    for i, file in enumerate(files):
        t, d = process_one(path / file.name)

        if ppm is None:
            ppm = d[:, 0]
            ppm_slice_index = slice(np.argmin(np.abs(ppm-ppm_slice.start)), np.argmin(np.abs(ppm-ppm_slice.stop)))
            ppm = ppm[ppm_slice_index]
            data = np.empty((len(files), len(ppm)), dtype=d.dtype)
        data[i, :] = d[ppm_slice_index, 1]
        times[i] = t
        print(i, ":", len(files), "done")

    return times, ppm, data


def pack_time_series(x: np.ndarray, time_: np.ndarray, z: np.array) -> np.ndarray:
    data = np.empty((len(time_) + 1, len(x) + 1), dtype=z.dtype)
    data[0, 0] = 0
    data[0, 1:] = x
    data[1:, 0] = time_
    data[1:, 1:] = z
    return data


def numpy_to_feather(array_: np.ndarray, file_path: str | pathlib.Path):
    """
    Save numpy array to feather file

    Parameters
    ----------
    array_:
        numpy array
    file_path:
        file path
    """
    if not isinstance(file_path, pathlib.Path):
        file_path = pathlib.Path(file_path)
    if file_path.suffix != ".feather":
        file_path = file_path.with_suffix(".feather")

    arrays = [pa.array(col) for col in array_]
    names = [str(i) for i in range(len(arrays))]
    batch = pa.RecordBatch.from_arrays(arrays, names=names)
    with pa.OSFile(str(file_path), 'wb') as sink:
        with pa.RecordBatchStreamWriter(sink, batch.schema) as writer:
            writer.write_batch(batch)


def main():
    path = pathlib.Path(r"C:\Users\nicep\Desktop\DW2-7")
    times, ppm, data = process_many(path)
    data = pack_time_series(ppm, times, data)

    # save
    # np.savetxt("DW2_5_1_NMR.csv", data, delimiter=",")
    numpy_to_feather(data, r"C:\Users\nicep\Desktop\DW2_flow_rate_NMR.feather")
    print("done")

import chem_analysis as ca

def load_many(path: pathlib.Path):
    files = tuple(os.scandir(path))
    files = sorted(files, key=lambda x: int(x.name[1:]))
    nmr_signals = []
    for file in files:
        nmr_ = ca.nmr.NMRSignal.from_spinsolve_csv(file.path)
        nmr_signals.append(nmr_)
        if len(nmr_signals) == 10:
            break

    return ca.nmr.NMRSignalArray.from_signals(nmr_signals)


def main2():
    path = pathlib.Path(r"C:\Users\nicep\Desktop\DW2-7")
    data = load_many(path)
    data.processor.add(ca.processing.re_sampling.CutSpans(x_spans=(-2, 12), invert=True))

    data.to_npy(r"C:\Users\nicep\Desktop\DW2-7\nmr_data.npy")


if __name__ == "__main__":
    main2()
