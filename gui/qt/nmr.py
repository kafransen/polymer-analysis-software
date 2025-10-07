
import numpy as np
from PyQt6 import QtWidgets 
from PyQt6 import QtCore
from PyQt6 import QtGui
import pyqtgraph as pg


pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')


class NMRView(QtWidgets.QWidget):
    def __init__(self, window: QtWidgets.QMainWindow):
        super().__init__()

        menubar = window.menuBar()
        NMRMenuBar(menubar)
        mainLayout = QtWidgets.QHBoxLayout()
        centralWidget = QtWidgets.QWidget()
        centralWidget.setLayout(mainLayout)
        self.setCentralWidget(centralWidget)

        # if model is not None:
        #     self.updateView()

    def updateView(self):
        widgetAll = QtWidgets.QWidget()
        widgetAllLayout = QtWidgets.QHBoxLayout()
        widgetAll.setLayout(widgetAllLayout)

        self.dataWidget = NmrViewWidget(model=self.model)

        self.processorWidget = ProcessorViewWidget(model=self.model)

        self.processorWidget.reprocessed.connect(self.dataWidget.update)

        widgetAllLayout.addWidget(self.dataWidget)
        widgetAllLayout.addWidget(self.processorWidget)

        self.setCentralWidget(widgetAll)

        self.show()

    def open(self, exampleData=False):
        # if exampleData:
        #     path = "/Users/benno/Dropbox/Software/pyNMR/examples/data/bruker/INADEQUATE/2/"
        #
        # else:
        #     path = QtWidgets.QFileDialog.getExistingDirectory(self, "Open Experiment", os.path.expanduser('~')) + "/"
        #
        #     print(path)
        #
        # data = topSpin.TopSpin(path)
        #
        # Processor = PROC.Processor([OPS.LeftShift(data.shiftPoints),
        #                             OPS.LineBroadening(0.0),
        #                             OPS.FourierTransform(),
        #                             OPS.Phase0D(-90),
        #                             OPS.Phase1D(data.timeShift,
        #                                         unit="time")])
        #
        # self.model = pyNmrDataModel(dataSet=pyNmrDataSet(data=data,
        #                                                  processor=Processor))

        self.updateView()


class NmrViewWidget(QtWidgets.QFrame):
    """A widget to display NMR data"""
    pivotChanged = QtCore.pyqtSignal()

    def __init__(self, model=None, dataSetIndex=0):

        super().__init__()
        self.pw = pg.PlotWidget(name="plot1")

        self.domain = None
        self.pivotPosition = 0
        self.showPivot = False

        # this is a plot data item
        self.p1 = self.pw.plot()
        self.p1.setPen((20, 20, 200))

        self.xLabel = "Time"
        self.xUnit = "s"
        self.pw.setXRange(0, 2)
        # pw.setYRange(0, 1e-10)

        # self.pivotLine =

        self.pPivot = pg.InfiniteLine(angle=90, movable=True)
        self.pPivot.setPen((200, 20, 20))

        self.pPivot.sigPositionChangeFinished.connect(self.pivotChanged)

        self.pw.addItem(self.pPivot)

        self.y, self.x = np.random.rand(100), np.random.rand(100)
        self.updatePW()

        layout = QtWidgets.QVBoxLayout()

        self.setLayout(layout)

        layout.addWidget(self.pw)

        self.model = model

        if model is not None and len(model.dataSets) > 0:
            self.update()
            self.pw.autoRange()

    def showPivotSignal(self, show):
        print("Setting Pivot to " + str(show))
        self.showPivot = show
        self.updatePW()

    def pivotPositionSignal(self, val):
        print("Updating pivot position to {}".format(val))

        self.pivotPosition = float(val)
        self.updatePW()

    def changeAxis(self, axis):
        print("Change Axis called in nmrView.")
        self.update(domain=axis)

    def reprocessed(self):
        self.update()

    def update(self, domain=None, position=-1, index=0, dataSetIndex=0):
        """Update plot.
        Optional keyword arguments:
        domain=None | "TIME" | "FREQUENCY" | "PPM"
        position=-1
        index=0
        dataSetIndex = 0

        If no domain is specified the plot will show
        FREQUENCY domain data at the last position and at index 0.
        """
        print("Datsetindex: ", dataSetIndex)
        print("Position: ", position)

        replot = False

        if domain is None and self.domain is None:
            if hasattr(self.model.dataSets[dataSetIndex].data, "ppmScale"):
                domain = "PPM"
                self.domain = "PPM"
                replot = True
            if len(self.model.dataSets[dataSetIndex].data.allSpectra) > 0:
                domain = "FREQUENCY"
            else:
                domain = "TIME"
        elif domain:
            self.domain = domain

        print("Domain: ", domain)

        if domain == "TIME":
            self.y = np.real(self.model.dataSets[dataSetIndex].data.allFid[position][index])
            self.x = self.model.dataSets[dataSetIndex].data.fidTime
            self.xLabel = "Time"
            self.xUnit = "s"
        elif domain == "FREQUENCY":
            self.y = np.real(self.model.dataSets[dataSetIndex].data.allSpectra[position][index])
            self.x = self.model.dataSets[dataSetIndex].data.frequency
            self.xLabel = "Frequency"
            self.xUnit = "Hz"
        elif domain == "PPM":
            self.y = np.real(self.model.dataSets[dataSetIndex].data.allSpectra[position][index])
            self.x = self.model.dataSets[dataSetIndex].data.ppmScale
            self.xLabel = "Chemical Shift"
            self.xUnit = "PPM"

        self.updatePW(replot=replot)

        # update the plots viewbox to show all data.
        self.pw.setMouseEnabled(x=True, y=True)

        if replot:
            print("Replotting")
            self.pw.autoRange()

        # self.pw.manualRange()

    # when Shift key is pressed, zoom y range as well.
    # for now you have to press shift as well.
    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key.Key_X:
            self.pw.setMouseEnabled(y=False)
        if event.key() == QtCore.Qt.Key.Key_Y or event.key() == QtCore.Qt.Key.Key_Z:
            self.pw.setMouseEnabled(x=False)
        if event.key() == QtCore.Qt.Key.Key_A:
            self.pw.autoRange()

    def keyReleaseEvent(self, event):
        if event.key() == QtCore.Qt.Key.Key_X:
            self.pw.setMouseEnabled(y=True)
        if event.key() == QtCore.Qt.Key.Key_Y or event.key() == QtCore.Qt.Key.Key_Z:
            self.pw.setMouseEnabled(x=True)

    def updatePW(self, replot=False):
        self.pw.setLabel('bottom', self.xLabel, units=self.xUnit)

        # self.pw.setXRange(0, 2)
        # pw.setYRange(0, 1e-10)
        self.p1.setData(y=self.y, x=self.x)

        # change this code to draw a proper line, and
        # change it to be shown or not.

        self.pPivot.setValue(self.pivotPosition)

        # if self.showPivot:
        #    self.pPivot.
        # self.pPivot.setData(y=[-1e9,1e10], x=[self.pivotPosition, self.pivotPosition])

        if self.domain == "PPM":
            self.p1.getViewBox().invertX(True)

        if replot:
            print("Replotting in PW")
            self.pw.autoRange()


##################################################################################################################
from functools import partial
import gui.qt.widgts.NMR as ow


class ProcessorViewWidget(QtWidgets .QFrame):
    """A wdiget to display NMR data"""
    reprocessed = QtCore.pyqtSignal()

    # this signal is emitted after Fourier Transform, and should
    # cause the axis to change to the ppm scale
    changeAxis = QtCore.pyqtSignal(str)

    # this will set the pivot
    pivotPositionSignal = QtCore.pyqtSignal(str)

    pivotPositionChange = QtCore.pyqtSignal()
    showPivotSignal = QtCore.pyqtSignal(int)

    def __init__(self, model=None, dataSetIndex=0, parent=None):

        super().__init__()
        self.model = model
        self.parent = parent

        layout = QtWidgets .QVBoxLayout()
        self.setLayout(layout)

        self.dataSetIndex = dataSetIndex
        pStack = self.model.dataSets[dataSetIndex].processorStack

        self.pWidgets = []

        # for every processor add a little frame
        for number, p in enumerate(pStack):
            runFunc = partial(self.runProcessor, p)

            pBox = QtWidgets .QGroupBox("Processor {}".format(number + 1))
            layout.addWidget(pBox, 1)

            thisProcessorLayout = QtWidgets .QVBoxLayout()
            pBox.setLayout(thisProcessorLayout)

            # add widgets for operatin
            # widgets that can initate a run of the processor get the run function as
            # an optional keyword argument. If they have it, they will run it,
            # and this function will run the processor within this object.
            for op in p:
                print("name: ", op.name)

                if op.name == "Left Shift":
                    self.pWidgets.append(ow.LeftShiftWidget(op))
                elif op.name == "Exponential Linebroadening":
                    self.pWidgets.append(ow.ExponentialLineBroadening(op))
                elif op.name == "Fourier Transform":
                    self.pWidgets.append(ow.FourierTransform(op))
                elif op.name == "Set PPM Scale":
                    self.pWidgets.append(ow.SetPPMScale(op, parent=self, runFunc=runFunc))
                    print("Emitting change Axis signal.")
                    self.changeAxis.emit("PPM")
                elif op.name == "Phase Zero Order":
                    self.pWidgets.append(ow.PhaseZeroOrder(op, runFunc=runFunc))
                elif op.name == "Phase First Order":
                    self.pWidgets.append(ow.PhaseFirstOrder(op, parent=self, runFunc=runFunc))
                    self.pWidgets[-1].showPivotSignal.connect(self.showPivotSignal)
                    self.pWidgets[-1].pivotPositionSignal.connect(self.pivotPositionSignal)
                    self.parent.dataWidget.pivotChanged.connect(self.pWidgets[-1].updatePivotPosition)

                thisProcessorLayout.addWidget(self.pWidgets[-1])

            # runButton = QtWidgets.QPushButton("Process (P)", self,
            #                             shortcut=QtGui.QKeySequence("P"),
            #                             clicked=partial(self.runProcessor, p))
            #
            # saveParametersButton = QtWidgets.QPushButton("Save Processor", self,
            #                                        clicked=self.saveProcessor)
            #
            # saveSpectrumButton = QtWidgets.QPushButton("Save Data", self, clicked=self.saveData)

            runButton = QtWidgets.QPushButton("Process (P)", self)
            runButton.setShortcut("P")
            runButton.clicked.connect(partial(self.runProcessor, p))

            saveParametersButton = QtWidgets.QPushButton("Save Processor", self)
            saveParametersButton.clicked.connect(self.saveProcessor)

            saveSpectrumButton = QtWidgets.QPushButton("Save Data", self)
            saveSpectrumButton.clicked.connect(self.saveData)

            thisProcessorLayout.addWidget(runButton)
            thisProcessorLayout.addWidget(saveParametersButton)
            thisProcessorLayout.addWidget(saveSpectrumButton)

    def saveProcessor(self):
        # for now we save processor 1 of this widgets dataset.
        pathToProcessorFile = self.model.dataSets[self.dataSetIndex].data.path + "pynmrProcessor1.pickle"
        # dump(self.model.dataSets[self.dataSetIndex].processorStack[0],
        #           file=open(pathToProcessorFile, "wb"))

    def saveData(self):
        self.model.dataSets[self.dataSetIndex].data.saveAscii()

    def test(self):
        print("HI")

    def runProcessor(self, processor):
        processor.runStack(self.model.dataSets[self.dataSetIndex].data)
        self.reprocessed.emit()
        self.changeAxis.emit("PPM")

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key.Key_P:
            print("F5 pressed. Run processor.")
            self.model.dataSets[0].processorStack[0].runStack(
                self.model.dataSets[self.dataSetIndex])


class NMRMenuBar:
    def __init__(self, window):
        menubar = window.menuBar()
        menu_file = menubar.addMenu("File")
        menu_processing = menubar.addMenu("Processing")
        menu_analysis = menubar.addMenu("Analysis")

        action_open = menu_file.addAction("Open", self.open)
        action_save = menu_file.addAction("Save")
        action_quit = menu_file.addAction("Quit", self.destroy)



        # Toolbars
        TBfile = window.addToolBar("File")
        TBfile.addAction(action_open)
        TBfile.addAction(action_save)

        open_icon = self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DirOpenIcon)
        save_icon = self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DialogSaveButton)

        action_open.setIcon(open_icon)
        action_save.setIcon(save_icon)

        TBfile.setAllowedAreas(QtCore.Qt.ToolBarArea.AllToolBarAreas)

        TBviewerNavigation = window.addToolBar("Viewer Navigation")

        action_previous = QtGui.QAction("<", self)
        action_next = QtGui.QAction(">", self)

        TBviewerNavigation.addAction(action_previous)
        TBviewerNavigation.addAction(action_next)
        TBviewerNavigation.setAllowedAreas(
            QtCore.Qt.ToolBarArea.TopToolBarArea | QtCore.Qt.ToolBarArea.BottomToolBarArea
        )