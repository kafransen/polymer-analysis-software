import abc

import numpy as np

from chem_analysis.processing.base import ProcessingMethod


class PhaseCorrection(ProcessingMethod, abc.ABC):

    @abc.abstractmethod
    def run(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ...


class Phase0D(PhaseCorrection):
    def __init__(self, phase, degree=True, domain="F"):
        """

        Parameters
        ----------
        phase:
            phase is in degree
        degree
        domain
        """
        self.phase = phase
        self.degree = degree
        self.domain = domain

    def run(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.degree:
            phaseFactor = np.exp(-1j*float(self.phase)/180.*np.pi)
        else:
            phaseFactor = np.exp(-1j*self.phase)

        if self.domain == "F":
            spectra = [spec*phaseFactor for spec in nmrData.allSpectra[-1]]
            nmrData.allSpectra.append(spectra)
        elif self.domain == "T":
            fids = [fid*phaseFactor for fid in nmrData.allFid[-1]]
            nmrData.allFid.append(fids)
        return x, y


class Phase1D(PhaseCorrection):
    def __init__(self, value, pivot=0, scale="Hz", unit="radian"):
        """

        Parameters
        ----------
        phase:
            phase is in degree
        degree
        domain
        """
        self.name = "Phase First Order"
        self.value = value
        self.pivot = pivot
        self.unit = unit
        self.scale = scale

    def run(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.unit == "radian":
            self.phase = self.value
        elif self.unit == "degree":
            self.phase = self.value * np.pi / 180
        elif self.unit == "time":
            self.phase = 2 * np.pi * nmrData.frequency[-1] * self.value

        print("Phase: ", self.phase)

        phaseValues = np.linspace(-self.phase / 2, self.phase / 2,
                                  num=len(nmrData.frequency))

        if self.pivot != 0:
            o = GetIndex(self.pivot, scale=self.scale)
            i = o.run(nmrData)
            phaseValues = phaseValues - phaseValues[i]

        spectra = [spec * np.exp(-1j * phaseValues)
                   for spec in nmrData.allSpectra[-1]]
        nmrData.allSpectra.append(spectra)
        return x, y

# class GetIndex(Operation):
#     def __init__(self, value, scale="Hz"):
#         """Get Indices corresponding to the frequency or ppm Value."""
#         self.value = value
#         self.scale = scale
#         self.name = "Get Index"
#
#     def run(self, nmrData):
#         if self.scale == "Hz":
#             index = np.argmin(abs(nmrData.frequency - self.value))
#         elif self.scale == "ppm":
#             index = np.argmin(abs(nmrData.ppmScale - self.value))
#
#         return index

