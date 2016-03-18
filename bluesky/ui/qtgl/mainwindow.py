try:
    from PyQt5.QtCore import Qt, pyqtSlot, QRect
    from PyQt5.QtGui import QPixmap, QIcon
    from PyQt5.QtWidgets import QMainWindow, QMenuBar, QMenu, QSplashScreen, QLabel, QWidget, QFrame
    from PyQt5 import uic
except ImportError:
    from PyQt4.QtCore import Qt, pyqtSlot, QRect
    from PyQt4.QtGui import QPixmap, QMainWindow, QMenuBar, QMenu, QIcon, QSplashScreen, QLabel, QWidget, QFrame
    from PyQt4 import uic

# Local imports
from uievents import PanZoomEvent


class Splash(QSplashScreen):
    """ Splash screen: BlueSky logo during start-up"""
    def __init__(self):
        super(Splash, self).__init__(QPixmap('data/graphics/splash.gif'), Qt.WindowStaysOnTopHint)


class MainWindow(QMainWindow):
    """ Qt window process: from .ui file read UI window-definitionof main window """

    def __init__(self, app, radarwidget):
        super(MainWindow, self).__init__()
        self.app = app
        uic.loadUi("./data/graphics/mainwindow.ui", self)

        # list of buttons to connect to, give icons, and tooltips
        #           the button         the icon      the tooltip    the callback
        buttons = { self.zoomin  :    ['zoomin.svg', 'Zoom in', self.buttonClicked],
                    self.zoomout :    ['zoomout.svg', 'Zoom out', self.buttonClicked],
                    self.panleft :    ['panleft.svg', 'Pan left', self.buttonClicked],
                    self.panright :   ['panright.svg', 'Pan right', self.buttonClicked],
                    self.panup :      ['panup.svg', 'Pan up', self.buttonClicked],
                    self.pandown :    ['pandown.svg', 'Pan down', self.buttonClicked],
                    self.ic :         ['stop.svg', 'Initial condition', self.buttonClicked],
                    self.op :         ['play.svg', 'Operate', self.buttonClicked],
                    self.hold :       ['pause.svg', 'Hold', self.buttonClicked],
                    self.fast :       ['fwd.svg', 'Enable fast-time', self.buttonClicked],
                    self.fast10 :     ['ffwd.svg', 'Fast-forward 10 seconds', self.buttonClicked],
                    self.sameic :     ['frwd.svg', 'Restart same IC', self.buttonClicked],
                    self.showac :     ['AC.svg', 'Show/hide aircraft', self.buttonClicked],
                    self.showpz :     ['PZ.svg', 'Show/hide PZ', self.buttonClicked],
                    self.showapt :    ['apt.svg', 'Show/hide airports', self.buttonClicked],
                    self.showwpt :    ['wpt.svg', 'Show/hide waypoints', self.buttonClicked],
                    self.showlabels : ['lbl.svg', 'Show/hide text labels', self.buttonClicked],
                    self.showmap :    ['geo.svg', 'Show/hide satellite image', self.buttonClicked]}

        self.simnodemenu = QMenu()
        node = self.simnodemenu.addAction('Node 1')
        node.setCheckable(True)
        node.setChecked(True)
        self.simnodemenu.addSeparator()
        addaction = self.simnodemenu.addAction('Add node')
        f = addaction.font()
        f.setBold(True)
        addaction.setFont(f)
        addaction.setEnabled(False)
        self.simnodes.setMenu(self.simnodemenu)
        for b in buttons.iteritems():
            # Set icon
            if not b[1][0] is None:
                icon = QIcon('data/graphics/icons/' + b[1][0])
                b[0].setIcon(icon)
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

        # Analysis and metrics menu
        self.analysisMenu = self.menubar.addMenu('&Analysis')
        # self.SD_action = self.analysisMenu.addAction('Static Density')
        # self.DD_action = self.analysisMenu.addAction('Dynamic Density')
        # self.SSD_action = self.analysisMenu.addAction('SSD Metric')
        # self.lyu_action = self.analysisMenu.addAction('Lyapunov analysis')

        # Connections menu
        self.connectionsMenu = self.menubar.addMenu('Connections')
        self.connectionsMenu.addAction('Connect to ADS-B server')
        self.connectionsMenu.addAction('Enable output to UDP')

        self.setMenuBar(self.menubar)

        self.radarwidget = radarwidget
        radarwidget.setParent(self.centralwidget)
        self.verticalLayout.insertWidget(0, radarwidget, 1)

    def closeEvent(self, event):
        self.app.closeAllWindows()

    @pyqtSlot()
    def buttonClicked(self):
        if self.sender() == self.zoomin:
            self.app.notify(self.app, PanZoomEvent(zoom=1.4142135623730951))
        elif self.sender() == self.zoomout:
            self.app.notify(self.app, PanZoomEvent(zoom=0.70710678118654746))
        elif self.sender() == self.pandown:
            self.app.notify(self.app, PanZoomEvent(pan=(-0.5,  0.0)))
        elif self.sender() == self.panup:
            self.app.notify(self.app, PanZoomEvent(pan=( 0.5,  0.0)))
        elif self.sender() == self.panleft:
            self.app.notify(self.app, PanZoomEvent(pan=( 0.0, -0.5)))
        elif self.sender() == self.panright:
            self.app.notify(self.app, PanZoomEvent(pan=( 0.0,  0.5)))
        elif self.sender() == self.ic:
            self.app.show_file_dialog()
        elif self.sender() == self.sameic:
            self.app.stack('IC IC')
        elif self.sender() == self.hold:
            self.app.stack('HOLD')
        elif self.sender() == self.op:
            self.app.stack('OP')
        elif self.sender() == self.fast:
            self.app.stack('FF')
        elif self.sender() == self.fast10:
            self.app.stack('FF 0:0:10')
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

