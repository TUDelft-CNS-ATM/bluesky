try:
    from collections.abc import Collection
except ImportError:
    # In python <3.3 collections.abc doesn't exist
    from collections import Collection
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
matplotlib.rcParams['font.size'] = 5

from matplotlib.figure import Figure
from matplotlib.backend_bases import key_press_handler
import matplotlib.pyplot as plt
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QTabWidget, QVBoxLayout, QScrollArea, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, \
    NavigationToolbar2QT as NavigationToolbar

import bluesky as bs


class InfoWindow(QTabWidget):
    ''' Top-level window containing simulation information such as plots. '''
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
    ''' InfoWindow tab for plots. '''
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
        ''' Update plots in this tab using incoming data. '''
        for fig, figdata in data.items():
            # First extract plot data when present
            x, y = figdata.pop('x', None), figdata.pop('y', None)

            plot = self.plots.get((sender, fig))
            if not plot:
                # If plot doesn't exist yet, create it
                plot = Plot(self, **figdata)
                self.plots[(sender, fig)] = plot
                self.layout.addWidget(plot)
            elif figdata:
                # When passed, apply updated figure settings
                plot.set(**figdata)

            if x is not None and y is not None:
                plot.update_data(x, y)

class Plot(FigureCanvas):
    def __init__(self, parent, plot_type='line', **kwargs):
        super(Plot, self).__init__(plt.figure())
        self.setParent(parent)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()
        self.setFixedHeight(350)
        self.axes = self.figure.add_subplot(111, **kwargs)
        self.figure.tight_layout(pad=8)
        self.plots = []
        self.plot_type = plot_type
        self.data = []

    def set(self, **kwargs):
        for flag, value in kwargs.items():
            if flag == 'legend':
                if len(self.plots) < len(value):
                    for _ in range(len(value) - len(self.plots)):
                        if self.plot_type == 'line':
                            lineobj = self.axes.plot(np.array([]), np.array([]))[0]
                        self.plots.append(lineobj)
                self.axes.legend(value)

    def update_data(self, xdata, ydata):
        if self.plot_type == 'line':
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
                    lineobj = self.axes.plot(np.array([]), np.array([]))[0]
                    self.plots.append(lineobj)

            for p, x, y in zip(self.plots, xdata, ydata):
                p.set_xdata(np.append(p.get_xdata(), x))
                p.set_ydata(np.append(p.get_ydata(), y))
                p.axes.relim()
                p.axes.autoscale_view()

        elif self.plot_type == 'boxplot' and len(ydata):
                nnewplots = len(ydata) - len(self.data)
                if nnewplots > 0:
                    self.data.extend(nnewplots * [[]])
                for i, d in enumerate(ydata):
                    self.data[i].extend(d)
                # self.data = [x.extend(n) for x, n in zip(self.data, ydata)]
                self.data = [d for d in self.data if d]
                if len(self.data):
                    self.axes.cla()
                    self.axes.boxplot(self.data)


        self.draw()
        self.flush_events()
