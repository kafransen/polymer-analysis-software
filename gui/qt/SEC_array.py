import pathlib
import os

import numpy as np
from PyQt6 import QtWidgets
from PyQt6 import QtCore
from PyQt6 import QtGui

import pyqtgraph as pg
from pyqtgraph.parametertree import interact, ParameterTree, Parameter
from pyqtgraph.dockarea.Dock import Dock
from pyqtgraph.dockarea.DockArea import DockArea
from pyqtgraph.Qt import QtCore


from chem_analysis.utils.feather_format import unpack_and_merge_time_series_feather_files, feather_to_numpy, \
    unpack_time_series


pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')


class Data:
    def __init__(self, retention_time: np.ndarray, times: np.ndarray, data: np.ndarray):
        self.time = times
        self.time_zeroed = self.time - self.time[0]
        self.retention_time = retention_time
        self.data = data

    @property
    def x(self):
        return self.retention_time


def load_example() -> IRData:
    x, y, z = unpack_and_merge_time_series_feather_files(
            [
                r"G:\Other computers\My Laptop\post_doc_2022\Data\polymerizations\DW2-7\DW2_7_flow_ATIR.feather",
                r"G:\Other computers\My Laptop\post_doc_2022\Data\polymerizations\DW2-7\DW2_7_flow_ATIR2.feather",
                r"G:\Other computers\My Laptop\post_doc_2022\Data\polymerizations\DW2-7\DW2_7_flow_ATIR3.feather"
            ]
        )
    return IRData(x, y, z)


class CustomViewBox(pg.ViewBox):
    def __init__(self, *args, **kwds):
        kwds['enableMenu'] = False
        pg.ViewBox.__init__(self, *args, **kwds)
        self.setMouseMode(self.RectMode)

    ## reimplement right-click to zoom out
    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.MouseButton.RightButton:
            self.autoRange()

    ## reimplement mouseDragEvent to disable continuous axis zoom
    def mouseDragEvent(self, ev, axis=None):
        if axis is not None and ev.button() == QtCore.Qt.MouseButton.RightButton:
            ev.ignore()
        else:
            pg.ViewBox.mouseDragEvent(self, ev, axis=axis)


class IRArrayView(QtWidgets.QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data: IRData = None  # noqa
        self.main_plot: pg.PlotWidget = None  # noqa
        self.time_plot: pg.PlotWidget = None  # noqa
        self.tree: ParameterTree = None  # noqa
        self._num_curves = None
        self._time_line = None
        self._main_plot_line = None
        self._plot_index = None
        self._tree_x = None
        self._tree_y = None

        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)
        self.build_menubar()
        self.stuff()

    def stuff(self):
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
        i3 = pg.TreeWidgetItem(["# Spectra"])
        self._num_curves = pg.SpinBox(value=5, int=True, dec=True, minStep=1, step=1, bounds=[1, 10])
        i3.setWidget(1, self._num_curves)
        self.tree.addTopLevelItem(i3)
        d3.addWidget(self.tree)

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

        x = self.data.time_zeroed
        y = np.trapz(y=self.data.data, axis=1)
        self.time_plot.plot(x, y, pen=(255, 255, 255, 200))
        self._time_line = pg.InfiniteLine(201, movable=True, bounds=(0, np.max(x)), label='x={value:0.2f}',
                             labelOpts={
                                 'position': 0.1, 'color': (200, 200, 100), 'fill': (200, 200, 200, 50), 'movable': True
                             }, name="line")
        self.time_plot.addItem(self._time_line)
        self.time_plot.setLimits(xMin=np.min(x), xMax=np.max(x))

        self._time_line.sigPositionChanged.connect(self.update_main_plot)
        self.update_main_plot()

    def update(self) -> None:
        self.update_time_plot()

    def build_menubar(self):
        window = self.parent().window()
        menubar = window.menuBar()

        # main file
        menu_file = menubar.addMenu("File")
        action_open = menu_file.addAction("Open", self.open)
        action_save = menu_file.addAction("Save", self.save)
        action_quit = menu_file.addAction("Quit", self.quit)

        # processing
        menu_processing = menubar.addMenu("Processing")
        action_smooth = menu_processing.addAction("Smooth", self.smooth)
        action_baseline = menu_processing.addAction("Baseline Correction", self.baseline)

        # analysis
        menu_analysis = menubar.addMenu("Analysis")
        action_integrate = menu_analysis.addAction("Integrate", self.integrate)

        # Toolbars
        TBfile = window.addToolBar("File")
        TBfile.addAction(action_open)
        TBfile.addAction(action_save)

        open_icon = window.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DirOpenIcon)
        save_icon = window.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DialogSaveButton)

        action_open.setIcon(open_icon)
        action_save.setIcon(save_icon)

        TBfile.setAllowedAreas(QtCore.Qt.ToolBarArea.TopToolBarArea)

    def open(self, exampleData=False):
        print("open")
        path = QtWidgets.QFileDialog.getOpenFileName(parent=self,
                                                     caption="Open feather file",
                                                     directory=os.path.expanduser('~'),
                                                     initialFilter="feather"
                                                     )
        print(path)
        self.data = IRData(*unpack_time_series(feather_to_numpy(path[0])))
        self.update()

    def save(self):
        print("save")

    def quit(self):
        print("quit")

    def smooth(self):
        print("smooth")

    def baseline(self):
        print("baseline")

    def integrate(self):
        print("integrate")
