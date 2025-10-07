
from PyQt6 import QtCore

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore


class CustomViewBox(pg.ViewBox):
    def __init__(self, *args, **kwds):
        kwds['enableMenu'] = False
        pg.ViewBox.__init__(self, *args, **kwds)
        self.setMouseMode(self.RectMode)

    def mouseClickEvent(self, ev):
        """reimplement right-click to zoom out"""
        if ev.button() == QtCore.Qt.MouseButton.RightButton:
            self.autoRange()

    def mouseDragEvent(self, ev, axis=None):
        """reimplement mouseDragEvent to disable continuous axis zoom"""
        if axis is not None and ev.button() == QtCore.Qt.MouseButton.RightButton:
            ev.ignore()
        else:
            pg.ViewBox.mouseDragEvent(self, ev, axis=axis)
