from __future__ import annotations
import enum
from datetime import datetime, timedelta

import numpy as np


class NMRExperiments(str, enum.Enum):
    UNKNOWN = "unknown"
    PROTON = "1H"
    TOCSY = "TOCSY"
    NOESY = "NOESY"
    COSY = "COSY"
    CARBON = "13C"
    DEPT = "DEPT"
    HETCOR = "HETCOR"
    HSQC = "HSQC"
    HSBC = "HSBC"
    FLUORINE = "19F"
    PHOSPHORUS = "31P"


class NMRParameters:
    _file_processors = dict()

    def __init__(self,
                 sample_name: str = None,
                 type_: NMRExperiments = None,
                 solvent: str = None,
                 date_start: datetime = None,
                 date_end: datetime = None,
                 spectrometer_frequency: float = None,  # Hz
                 temperature: float = None,  # Kelvin

                 number_scans: int = None,
                 pulse_width: timedelta = None,  # range micro seconds
                 repetition_delay: timedelta = None,
                 acquisition_time: timedelta = None,
                 pulse_angle: float = None,  # degree
                 number_points: int = None,  # spectral_size
                 ):
        """
        type_:
            Type of NMR experiment
        solvent:
            solvent used in NMR
        date_start:
            date NMR experiment was started
        spectrometer_frequency:

        temperature:

        number_scans:

        pulse_width:

        repetition_delay:
            Recycle Delay or d1 (Varian)
            dwell time or DT
            relaxation delay D1
            scan delay
            is the time interval between each data point collected in the FID.
        acquisition_time:
            acquisition period or AQ
            The acquisition time, often denoted as AT, is the total duration for which the FID signal is recorded.
            It determines the length of the time domain signal. AT is usually set by the user and depends on the desired
            spectral width and resolution of the NMR spectrum.
        pulse_angle:

        number_points:
            The number of data points or data values collected during the FID acquisition is determined by the user and
            should be specified before the experiment. It affects the digital resolution and sensitivity of the resulting
            NMR spectrum. The greater the number of data points, the finer the spectral resolution, but the longer the
            acquisition time.
         """
        self.sample_name = sample_name
        self.type_ = type_
        self.solvent = solvent
        self.date_start = date_start
        self.date_end = date_end
        self.spectrometer_frequency = spectrometer_frequency
        self.temperature = temperature
        self.number_scans = number_scans
        self.pulse_width = pulse_width
        self.repetition_delay = repetition_delay
        self.acquisition_time = acquisition_time
        self.pulse_angle = pulse_angle
        self.number_points = number_points

    # instrument: str = None
    # pulse_sequence: str = None
    # probe: str = None
    # receiver_gain: float = None
    # shift_points: int = None
    # pulse_length: float = None

    def compute_time(self) -> np.ndarray:
        return np.linspace(0, (self.number_points - 1) * self.repetition_delay.total_seconds(), self.number_points)

    @property
    def sweep_width_Hz(self) -> float:
        """
        spectral width in ppm is independent of the spectrometer operating frequency; however, since the number of Hz
        per ppm is dependent on the spectrometer operating frequency, the spectral width in Hz will change depending
        upon the spectrometer used.

        """
        return self.number_points / 2 / self.acquisition_time.total_seconds()

    @property
    def sweep_width_ppm(self) -> float:
        """
        spectral width in ppm is independent of the spectrometer operating frequency; however, since the number of Hz
        per ppm is dependent on the spectrometer operating frequency, the spectral width in Hz will change depending
        upon the spectrometer used.

        """
        return self.sweep_width_Hz / self.spectrometer_frequency

    @property
    def resolution(self) -> float:
        """
        digital resolution of the spectrum
        The digital resolution is in units of Hz/point, and the rule-of-thumb is that the digital resolution (in Hertz)
        should be less than one half the natural peak width at half-height.
        This ensures that each peak is described by at least 3 points.
        """
        return 1 / self.acquisition_time.total_seconds()

    @property
    def total_scan_time(self) -> timedelta:
        return self.pulse_width + self.acquisition_time + self.repetition_delay

    @property
    def experiment_time(self) -> timedelta:
        return self.number_scans * self.total_scan_time


class NMRParameters2D(NMRParameters):
    pass


"""
Zero Filling (ZF): Zero filling is a processing step performed after data acquisition to increase the number of data points in the FID. Zero filling involves interpolating zeros between the acquired data points to increase the apparent resolution. It can be used to enhance the resolution without changing the acquisition time.

    Apodization Function: Apodization is a mathematical operation applied to the FID to modify its shape and reduce sideband artifacts in the Fourier-transformed spectrum. Common apodization functions include exponential multiplication, Gaussian multiplication, and sine-bell functions. The choice of apodization function affects the line shape and resolution of the resulting spectrum.

    Frequency Offset (O1): The frequency offset (O1) represents the position of the spectral center relative to the transmitter frequency. It is set according to the region of interest in the NMR spectrum. The frequency offset can be applied during the acquisition to shift the FID to the desired spectral region.
"""
