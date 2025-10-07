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

import chem_analysis as ca
from chem_analysis.plotting.qt_array import ArrayView


class IRArrayView(ArrayView):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.build_menubar()

    def build_menubar(self):
        window = self.parent.window()
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
                                                     # filter="feather",
                                                     initialFilter="feather"
                                                     )
        print(path)
        self.data = ca.ir.IRSignalArray.from_file(path[0])
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
