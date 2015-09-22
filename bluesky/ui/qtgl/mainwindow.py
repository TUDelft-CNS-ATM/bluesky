try:
    from PyQt4.QtCore import Qt, QRect, QPoint, QTimer, pyqtSlot
    from PyQt4.QtGui import QImage, QPixmap, QPainter, QColor, QMainWindow, QMenuBar, QBrush, QIcon, QPolygon, QLabel, QSplashScreen
    from PyQt4 import uic
except ImportError:
    from PyQt5.QtCore import Qt, QRect, QPoint, QTimer, pyqtSlot
    from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QBrush, QIcon, QPolygon
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
        img = QImage(32, 32, QImage.Format_ARGB32)
        painter = QPainter(img)
        painter.setBrush(QBrush(QColor(149, 179, 215)))
        img.fill(0)
        painter.drawRect(QRect(6, 6, 20, 20))
        self.ic.setIcon(QIcon(QPixmap.fromImage(img)))
        self.ic.setToolTip('Initial condition')
        img.fill(0)
        painter.drawRect(QRect(6, 6, 7, 20))
        painter.drawRect(QRect(19, 6, 7, 20))
        self.hold.setIcon(QIcon(QPixmap.fromImage(img)))
        self.hold.setToolTip('Hold')
        img.fill(0)
        painter.drawPolygon(QPolygon([QPoint(6, 6), QPoint(6, 26), QPoint(26, 16)]))
        self.op.setIcon(QIcon(QPixmap.fromImage(img)))
        self.op.setToolTip('Operate')
        painter.drawRect(QRect(6, 6, 4, 20))
        painter.drawPolygon(QPolygon([QPoint(9, 16), QPoint(18, 26), QPoint(18, 6)]))
        painter.drawPolygon(QPolygon([QPoint(17, 16), QPoint(26, 26), QPoint(26, 6)]))
        self.sameic.setIcon(QIcon(QPixmap.fromImage(img)))
        self.sameic.setToolTip('Restart same IC')
        img.fill(0)
        painter.drawPolygon(QPolygon([QPoint(6, 6), QPoint(6, 26), QPoint(16, 16)]))
        painter.drawPolygon(QPolygon([QPoint(16, 6), QPoint(16, 26), QPoint(26, 16)]))
        self.fast.setIcon(QIcon(QPixmap.fromImage(img)))
        self.fast.setToolTip('Enable fast-time')
        img.fill(0)
        painter.drawPolygon(QPolygon([QPoint(6, 6), QPoint(6, 26), QPoint(15, 16)]))
        painter.drawPolygon(QPolygon([QPoint(14, 6), QPoint(14, 26), QPoint(23, 16)]))
        painter.drawRect(QRect(22, 6, 4, 20))
        self.fast10.setIcon(QIcon(QPixmap.fromImage(img)))
        painter.end()

        self.ic.clicked.connect(self.buttonClicked)
        self.hold.clicked.connect(self.buttonClicked)
        self.op.clicked.connect(self.buttonClicked)
        self.sameic.clicked.connect(self.buttonClicked)
        self.fast.clicked.connect(self.buttonClicked)
        self.fast10.clicked.connect(self.buttonClicked)

        self.menubar = QMenuBar()

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

        # Siminfo label
        self.siminfoLabel = QLabel('F = 0 Hz')
        self.verticalLayout.addWidget(self.siminfoLabel)

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
