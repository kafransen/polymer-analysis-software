
import numpy as np

from PyQt6 import QtWidgets
from PyQt6 import QtCore
from PyQt6 import QtGui

import pyqtgraph as pg
from pyqtgraph.parametertree import interact, ParameterTree, Parameter
from pyqtgraph.dockarea.Dock import Dock
from pyqtgraph.dockarea.DockArea import DockArea
from pyqtgraph.Qt import QtCore

from chem_analysis.plotting.qt_helpers import CustomViewBox
from chem_analysis.base_obj.signal_array import SignalArray

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')


def qt_array(array_):
    """ Blocking code """
    app = QtWidgets.QApplication([])
    window = ArrayView(array_)
    window.update()
    window.show()
    # Set up a timer to close the window after a delay
    timer = QtCore.QTimer(window)

    def closeWindow():
        timer.stop()
        window.close()

    timer.timeout.connect(closeWindow)
    timer.start(10_000_000)  # Time in milliseconds
    app.exec()


class ArrayView(QtWidgets.QWidget):
    def __init__(self, data: SignalArray | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = data

        self.main_plot: pg.PlotWidget = None  # noqa
        self.time_plot: pg.PlotWidget = None  # noqa
        self.tree: ParameterTree = None  # noqa
        self._num_curves = None
        self._time_line = None
        self._main_plot_line = None
        self._plot_index = None
        self._tree_x = None
        self._tree_y = None
        self._tree_spectra_index = None

        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)
        self.build()

    def build(self):
        area = DockArea()
        area.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        d1 = Dock("Dock1", size=(800, 400), hideTitle=True)
        d2 = Dock("Dock2 - Console", size=(800, 100), hideTitle=True)
        d3 = Dock("Dock3", size=(200, 500), hideTitle=True)
        area.addDock(d1, 'left')
        area.addDock(d2, 'bottom')
        area.addDock(d3, 'right')
        self.layout().addWidget(area)

        vb = CustomViewBox()
        self.main_plot = pg.PlotWidget(viewBox=vb)
        self.main_plot.setLabels(left='absorption', bottom='wavenumber (cm-1)')
        d1.addWidget(self.main_plot)

        self.time_plot = pg.PlotWidget()
        self.time_plot.setLabels(left='signal', bottom='time (sec)')
        d2.addWidget(self.time_plot)

        self.tree = ParameterTree()
        self.tree.setMinimumWidth(150)
        d3.addWidget(self.tree)

        # number spectra showsn
        i3 = pg.TreeWidgetItem(["# spectra"])
        self._num_curves = pg.SpinBox(value=5, int=True, dec=True, minStep=1, step=1, bounds=[1, 10])
        i3.setWidget(1, self._num_curves)
        self.tree.addTopLevelItem(i3)

        # add index to tree
        i4 = pg.TreeWidgetItem(["spectra index"])
        self._tree_spectra_index = pg.ValueLabel(formatStr='{value:5}')
        i4.setWidget(1, self._tree_spectra_index)
        self.tree.addTopLevelItem(i4)

        self._num_curves.sigValueChanged.connect(self.update_main_plot)

    def update_main_plot(self):
        if self.data is None:
            return

        num_curves = self._num_curves.value()
        time_line_pos = self._time_line.value()
        index = np.argmin(np.abs(self.data.time_zeroed - time_line_pos))

        if index + num_curves <= len(self.data.time):
            start = index
            end = index + num_curves
        else:
            start = len(self.data.time) - num_curves
            end = len(self.data.time)

        self._plot_index = start
        self._tree_spectra_index.setValue(start)

        self.main_plot.clearPlots()
        for i in range(start, end):
            color = pg.intColor(i, num_curves)
            self.main_plot.plot(self.data.x, self.data.data[i, :], pen=color, name=f"Curve {i}")

        if self._main_plot_line is None:
            self._main_plot_line = pg.InfiniteLine(pos=np.mean(self.data.x), movable=True)
            self.main_plot.addItem(self._main_plot_line)
            i4 = pg.TreeWidgetItem(["x"])
            self._tree_x = pg.ValueLabel(formatStr='{value:4.0f}')
            i4.setWidget(1, self._tree_x)
            self.tree.addTopLevelItem(i4)
            i5 = pg.TreeWidgetItem(["y"])
            self._tree_y = pg.ValueLabel(formatStr='{value:1.4f}')
            i5.setWidget(1, self._tree_y)
            self.tree.addTopLevelItem(i5)

            self._main_plot_line.sigPositionChanged.connect(self.update_main_plot_line)
            self._time_line.sigPositionChanged.connect(self.update_main_plot_line)

    def update_main_plot_line(self):
        x = self._main_plot_line.value()
        self._tree_x.setValue(x)
        index = np.argmin(np.abs(self.data.x - x))
        y = self.data.data[self._plot_index, index]
        self._tree_y.setValue(y)

    def update_time_plot(self):
        if self.data is None:
            return

        self.create_time_plot()
        self.update_main_plot()

    def create_time_plot(self):
        x = self.data.time_zeroed
        y = np.trapz(y=self.data.data, axis=1)
        self.time_plot.plot(x, y, pen=(255, 0, 255, 200))
        self._time_line = pg.InfiniteLine(201, movable=True, bounds=(0, np.max(x)), label='x={value:0.2f}',
                                          labelOpts={
                                              'position': 0.1, 'color': (200, 200, 100), 'fill': (200, 200, 200, 50),
                                              'movable': True
                                          }, name="line")

        self.time_plot.addItem(self._time_line)
        self.time_plot.setLimits(xMin=np.min(x), xMax=np.max(x))

        self._time_line.sigPositionChanged.connect(self.update_main_plot)

    def update(self) -> None:
        self.update_time_plot()
