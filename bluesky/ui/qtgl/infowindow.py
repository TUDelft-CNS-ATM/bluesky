from collections.abc import Collection
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
matplotlib.rcParams['font.size'] = 5

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

import bluesky as bs


class InfoWindow(QTabWidget):
    def __init__(self):
        super(InfoWindow, self).__init__()
        self.setDocumentMode(True)
        self.resize(600, 500)
        self.plottab = None

        # Connect to sim data events
        bs.net.stream_received.connect(self.on_simstream_received)

    def add_plot_tab(self):
        self.plottab = PlotTab()
        self.addTab(self.plottab, 'Graphs')

    def on_simstream_received(self, streamname, data, sender_id):
        if streamname[:4] != b'PLOT':
            return

        if not self.plottab:
            self.add_plot_tab()
            self.show()

        self.plottab.update_plots(data, sender_id)

class PlotTab(QScrollArea):
    def __init__(self):
        super(PlotTab, self).__init__()
        self.layout = QVBoxLayout()
        container = QWidget()
        container.setLayout(self.layout)
        self.layout.setAlignment(Qt.AlignTop)
        self.setWidget(container)
        self.setWidgetResizable(True)
        self.plots = dict()

    def update_plots(self, data, sender):
        for fig, (x, y, color) in data.items():
            plot = self.plots.get((sender, fig))
            if not plot:
                plot = Plot(self)
                self.plots[(sender, fig)] = plot
                self.layout.addWidget(plot)

            plot.update_data(x, y)

class Plot(FigureCanvas):
    def __init__(self, parent):
        super(Plot, self).__init__(plt.figure())
        self.setParent(parent)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()
        self.setFixedHeight(350)
        self.plots = []

    def update_data(self, xdata, ydata, color=''):
        if isinstance(xdata, Collection) and not isinstance(ydata, Collection):
            ydata = [ydata] * len(xdata)
        elif not isinstance(xdata, Collection) and isinstance(ydata, Collection):
            xdata = [xdata] * len(ydata)
        elif not isinstance(xdata, Collection) and not isinstance(ydata, Collection):
            xdata = [xdata]
            ydata = [ydata]

        npoints = len(xdata)
        if len(self.plots) < npoints:
            for _ in range(npoints - len(self.plots)):
                self.plots.append(plt.plot(np.array([]), [], figure=self.figure)[0])

        for p, x, y in zip(self.plots, xdata, ydata):
            p.set_xdata(np.append(p.get_xdata(), x))
            p.set_ydata(np.append(p.get_ydata(), y))
            p.axes.relim()
            p.axes.autoscale_view()
            self.draw()
            self.flush_events()
