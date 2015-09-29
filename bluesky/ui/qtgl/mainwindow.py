try:
    from PyQt4.QtCore import Qt, QTimer, pyqtSlot
    from PyQt4.QtGui import QPixmap, QMainWindow, QMenuBar, QIcon, QLabel, QSplashScreen
    from PyQt4 import uic
except ImportError:
    from PyQt5.QtCore import Qt, QTimer, pyqtSlot
    from PyQt5.QtGui import QPixmap, QIcon
    from PyQt5.QtWidgets import QMainWindow, QMenuBar, QLabel, QSplashScreen
    from PyQt5 import uic


class Splash(QSplashScreen):
    def __init__(self):
        super(Splash, self).__init__(QPixmap('data/graphics/splash.gif'), Qt.WindowStaysOnTopHint)


class MainWindow(QMainWindow):

    def __init__(self, app, radarwidget):
        super(MainWindow, self).__init__()
        self.app = app
        uic.loadUi("./data/graphics/mainwindow.ui", self)

        # Tab 1
        self.ic.setIcon(QIcon(QPixmap('data/graphics/icons/stop.png')))
        self.ic.setToolTip('Initial condition')
        self.op.setIcon(QIcon(QPixmap('data/graphics/icons/play.png')))
        self.op.setToolTip('Operate')
        self.hold.setIcon(QIcon(QPixmap('data/graphics/icons/pause.png')))
        self.hold.setToolTip('Hold')
        self.fast.setIcon(QIcon(QPixmap('data/graphics/icons/fwd.png')))
        self.fast.setToolTip('Enable fast-time')
        self.fast10.setIcon(QIcon(QPixmap('data/graphics/icons/ffwd.png')))
        self.sameic.setIcon(QIcon(QPixmap('data/graphics/icons/frwd.png')))
        self.sameic.setToolTip('Restart same IC')

        # Tab 2
        self.showac.setIcon(QIcon(QPixmap('data/graphics/icons/AC.png')))
        self.showpz.setIcon(QIcon(QPixmap('data/graphics/icons/PZ.png')))
        self.showapt.setIcon(QIcon(QPixmap('data/graphics/icons/apt.png')))
        self.showwpt.setIcon(QIcon(QPixmap('data/graphics/icons/wpt.png')))
        self.showlabels.setIcon(QIcon(QPixmap('data/graphics/icons/lbl.png')))
        self.showmap.setIcon(QIcon(QPixmap('data/graphics/icons/geo.png')))

        self.ic.clicked.connect(self.buttonClicked)
        self.hold.clicked.connect(self.buttonClicked)
        self.op.clicked.connect(self.buttonClicked)
        self.sameic.clicked.connect(self.buttonClicked)
        self.fast.clicked.connect(self.buttonClicked)
        self.fast10.clicked.connect(self.buttonClicked)
        self.showac.clicked.connect(self.buttonClicked)
        self.showpz.clicked.connect(self.buttonClicked)
        self.showapt.clicked.connect(self.buttonClicked)
        self.showwpt.clicked.connect(self.buttonClicked)
        self.showlabels.clicked.connect(self.buttonClicked)
        self.showmap.clicked.connect(self.buttonClicked)

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
        self.siminfoLabel = QLabel('F = 0 Hz')
        self.verticalLayout.addWidget(self.siminfoLabel)

        self.radarwidget = radarwidget
        radarwidget.setParent(self.centralwidget)
        self.verticalLayout.insertWidget(0, radarwidget, 1)

        timer = QTimer(self)
        timer.timeout.connect(radarwidget.updateGL)
        timer.start(50)

    @pyqtSlot()
    def buttonClicked(self):
        if self.sender() == self.ic:
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
