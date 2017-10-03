import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
matplotlib.rcParams['backend.qt5'] = 'PyQt5'
from matplotlib.figure import Figure
from matplotlib.backend_bases import key_press_handler
import matplotlib.pyplot as plt
try:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QTabWidget, QVBoxLayout, QScrollArea, QWidget
    # from matplotlib.backends.backend_qt5 import FigureCanvas
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, \
        NavigationToolbar2QT as NavigationToolbar
except ImportError:
    from PyQt4.QtCore import Qt
    from PyQt4.QtGui import QTabWidget, QVBoxLayout, QScrollArea, QWidget

    from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas, \
        NavigationToolbar2QT as NavigationToolbar

class InfoWindow(QTabWidget):
    def __init__(self):
        super(InfoWindow, self).__init__()
        self.setDocumentMode(True)
        self.resize(600, 500)
        self.plottab = PlotTab()
        self.addPlotTab()

    def addPlotTab(self):
        self.addTab(self.plottab, 'Graphs')


class PlotTab(QScrollArea):
    def __init__(self):
        super(PlotTab, self).__init__()
        self.layout = QVBoxLayout()
        container = QWidget()
        container.setLayout(self.layout)
        self.layout.setAlignment(Qt.AlignTop)
        self.setWidget(container)
        self.setWidgetResizable(True)

    def addPlot(self):
        self.layout.addWidget(Plot(self))

class Plot(FigureCanvas):
    def __init__(self, parent):
        super(Plot, self).__init__(plt.figure())
        self.setParent(parent)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()
        self.setFixedHeight(250)
        self.plot = plt.plot(np.array([]), [], figure=self.figure)[0]

    def update_data(self, new_data):
        self.plot.set_xdata(np.append(self.plot.get_xdata(), new_data))
        self.plot.set_ydata(np.append(self.plot.get_ydata(), new_data))
        self.plot.axes.relim()
        self.plot.axes.autoscale_view()
        self.draw()
        self.flush_events()
