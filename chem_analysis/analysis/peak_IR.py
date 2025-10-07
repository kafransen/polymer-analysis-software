import dataclasses
import functools
from collections import OrderedDict

import numpy as np

from chem_analysis.analysis.peak import PeakBounded, PeakStats, PeakParent


@dataclasses.dataclass
class PeakParentIR(PeakParent):
    cm_1: np.ndarray


class PeakIR(PeakBounded):
    def __init__(self, parent: PeakParentIR, bounds: slice, id_: int = None):
        super().__init__(parent, bounds, id_)
        self.stats = PeakStats(self)
