
from chem_analysis.base_obj.calibration import Calibration
from chem_analysis.base_obj.chromatogram import Chromatogram
from chem_analysis.sec.sec_signal import SECSignal


class SECChromatogram(Chromatogram):

    def __init__(self, data:  list[SECSignal], calibration: Calibration = None):
        self.calibration = calibration

        super().__init__(data)
