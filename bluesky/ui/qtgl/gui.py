""" QTGL Gui for BlueSky."""
import os
import sys

from PyQt6.QtCore import QTimer, qInstallMessageHandler, QT_VERSION_STR

from PyQt6.QtCore import QtMsgType
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QFont

import bluesky as bs
from bluesky.network.client import Client
from bluesky.ui.qtgl.mainwindow import MainWindow, Splash, DiscoveryDialog

bs.settings.set_variable_defaults(qt_verbosity=1)

print(('Using Qt ' + QT_VERSION_STR + ' for windows and widgets'))

def gui_msg_handler(msgtype, context, msg):
    if msgtype == QtMsgType.QtDebugMsg and bs.settings.qt_verbosity > 3:
        print('Qt debug message:', msg)
    elif msgtype == QtMsgType.QtInfoMsg and bs.settings.qt_verbosity > 2:
        print('Qt information message:', msg)
    elif msgtype == QtMsgType.QtWarningMsg and bs.settings.qt_verbosity > 1:
        print('Qt gui warning:', msg)
    elif msgtype == QtMsgType.QtCriticalMsg and bs.settings.qt_verbosity > 0:
        print('Qt gui critical error:', msg)
    elif msgtype == QtMsgType.QtFatalMsg:
        print('Qt gui fatal error:', msg)
    


def start(hostname=None):
    # Install message handler for Qt messages
    qInstallMessageHandler(gui_msg_handler)

    # Avoid Window hidpi scaling of fonts affecting view of commands
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    #QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

    # Start the Qt main object
    app = QApplication(sys.argv[:1])

    # Explicitly set font to avoid font loading warning dialogs
    app.setFont(QFont('Sans'))

    # Start the bluesky network client
    client = Client()
    network_timer = QTimer()
    network_timer.timeout.connect(client.update)
    network_timer.start(20)

    splash = Splash()
    splash.show()

    splash.showMessage('Constructing main window')
    app.processEvents()
    win = MainWindow(bs.mode)
    win.show()
    splash.showMessage('Done!')
    app.processEvents()
    splash.finish(win)
    # If this instance of the gui is started in client-only mode, show
    # server selection dialog
    if bs.mode == 'client' and hostname is None:
        dialog = DiscoveryDialog(client.node_id, win)
        dialog.show()

    else:
        client.connect(hostname=hostname)

    # Start the Qt main loop
    # app.exec_()
    app.exec()