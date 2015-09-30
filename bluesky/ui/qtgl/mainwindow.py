try:
    from PyQt4.QtCore import Qt, QTimer, pyqtSlot
    from PyQt4.QtGui import QPixmap, QMainWindow, QMenuBar, QIcon, QSplashScreen
    from PyQt4 import uic
except ImportError:
    from PyQt5.QtCore import Qt, QTimer, pyqtSlot
    from PyQt5.QtGui import QPixmap, QIcon
    from PyQt5.QtWidgets import QMainWindow, QMenuBar, QSplashScreen
    from PyQt5 import uic

# Local imports
from uievents import PanZoomEvent


class Splash(QSplashScreen):
    def __init__(self):
        super(Splash, self).__init__(QPixmap('data/graphics/splash.gif'), Qt.WindowStaysOnTopHint)


class MainWindow(QMainWindow):

    def __init__(self, app, radarwidget):
        super(MainWindow, self).__init__()
        self.app = app
        uic.loadUi("./data/graphics/mainwindow.ui", self)

        # list of buttons to connect to, give icons, and tooltips
        #           the button         the icon      the tooltip    the callback
        buttons = { self.zoomin  :    ['zoomin.png', 'Zoom in', self.buttonClicked],
                    self.zoomout :    ['zoomout.png', 'Zoom out', self.buttonClicked],
                    self.ic :         ['stop.png', 'Initial condition', self.buttonClicked],
                    self.op :         ['play.png', 'Operate', self.buttonClicked],
                    self.hold :       ['pause.png', 'Hold', self.buttonClicked],
                    self.fast :       ['fwd.png', 'Enable fast-time', self.buttonClicked],
                    self.fast10 :     ['ffwd.png', 'Fast-forward 10 seconds', self.buttonClicked],
                    self.sameic :     ['frwd.png', 'Restart same IC', self.buttonClicked],
                    self.showac :     ['AC.png', 'Show/hide aircraft', self.buttonClicked],
                    self.showpz :     ['PZ.png', 'Show/hide PZ', self.buttonClicked],
                    self.showapt :    ['apt.png', 'Show/hide airports', self.buttonClicked],
                    self.showwpt :    ['wpt.png', 'Show/hide waypoints', self.buttonClicked],
                    self.showlabels : ['lbl.png', 'Show/hide text labels', self.buttonClicked],
                    self.showmap :    ['geo.png', 'Show/hide satellite image', self.buttonClicked]}

        for b in buttons.iteritems():
            # Set icon
            if not b[1][0] is None:
                b[0].setIcon(QIcon(QPixmap('data/graphics/icons/' + b[1][0])))
            # Set tooltip
            if not b[1][1] is None:
                b[0].setToolTip(b[1][1])
            # Connect clicked signal
            b[0].clicked.connect(b[1][2])

        self.menubar = QMenuBar(self)

        # File menu
        self.fileMenu = self.menubar.addMenu('&File')
        self.open_action = self.fileMenu.addAction('&Open')
        self.open_action.triggered.connect(self.app.show_file_dialog)
        self.save_action = self.fileMenu.addAction('&Save')

        # View Menu
        self.viewMenu = self.menubar.addMenu('&View')
        self.resetview_action = self.viewMenu.addAction('&Reset view')
        self.fullscreen_action = self.viewMenu.addAction('Fullscreen')

        # Analysis and metrics menu
        self.analysisMenu = self.menubar.addMenu('&Analysis')
        self.SD_action = self.analysisMenu.addAction('Static Density')
        self.DD_action = self.analysisMenu.addAction('Dynamic Density')
        self.SSD_action = self.analysisMenu.addAction('SSD Metric')
        self.lyu_action = self.analysisMenu.addAction('Lyapunov analysis')

        # Connections menu
        self.connectionsMenu = self.menubar.addMenu('Connections')
        self.connectionsMenu.addAction('Connect to ADS-B server')
        self.connectionsMenu.addAction('Enable output to UDP')

        self.setMenuBar(self.menubar)

        # Siminfo label
        # self.siminfoLabel = QLabel('F = 0 Hz')
        # self.verticalLayout.addWidget(self.siminfoLabel)

        self.radarwidget = radarwidget
        radarwidget.setParent(self.centralwidget)
        self.verticalLayout.insertWidget(0, radarwidget, 1)

        timer = QTimer(self)
        timer.timeout.connect(radarwidget.updateGL)
        timer.start(50)

    @pyqtSlot()
    def buttonClicked(self):
        if self.sender() == self.zoomin:
            self.app.notify(self.app, PanZoomEvent(PanZoomEvent.Zoom, 1.4142135623730951))
        elif self.sender() == self.zoomout:
            self.app.notify(self.app, PanZoomEvent(PanZoomEvent.Zoom, 0.70710678118654746))
        elif self.sender() == self.ic:
            self.app.show_file_dialog()
        elif self.sender() == self.sameic:
            self.app.stack('IC IC')
        elif self.sender() == self.hold:
            self.app.stack('HOLD')
        elif self.sender() == self.op:
            self.app.stack('OP')
        elif self.sender() == self.fast:
            print('Fast clicked')
        elif self.sender() == self.fast10:
            self.app.stack('RUNFT')
        elif self.sender() == self.showac:
            self.radarwidget.show_traf = not self.radarwidget.show_traf
        elif self.sender() == self.showpz:
            self.radarwidget.show_pz = not self.radarwidget.show_pz
        elif self.sender() == self.showapt:
            self.radarwidget.show_apt = not self.radarwidget.show_apt
        elif self.sender() == self.showwpt:
            self.radarwidget.show_wpt = not self.radarwidget.show_wpt
        elif self.sender() == self.showlabels:
            self.radarwidget.show_lbl = not self.radarwidget.show_lbl
        elif self.sender() == self.showmap:
            self.radarwidget.show_map = not self.radarwidget.show_map
