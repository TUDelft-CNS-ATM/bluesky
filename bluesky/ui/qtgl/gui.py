""" QTGL Gui for BlueSky."""
try:
    from PyQt5.QtCore import Qt, QEvent, qInstallMessageHandler, \
        QtMsgType, QT_VERSION, QT_VERSION_STR
    from PyQt5.QtWidgets import QApplication, QErrorMessage
    from PyQt5.QtGui import QFont
    
except ImportError:
    from PyQt6.QtCore import Qt, QEvent, qInstallMessageHandler, \
        QT_VERSION, QT_VERSION_STR

    from PyQt6.QtCore import QtMsgType
    from PyQt6.QtWidgets import QApplication, QErrorMessage
    from PyQt6.QtGui import QFont

import bluesky as bs
from bluesky.ui.qtgl.guiclient import GuiClient
from bluesky.ui.qtgl.mainwindow import MainWindow, Splash, DiscoveryDialog
from bluesky.ui.qtgl.customevents import NUMCUSTOMEVENTS


print(('Using Qt ' + QT_VERSION_STR + ' for windows and widgets'))

def gui_msg_handler(msgtype, context, msg):
    if msgtype == QtMsgType.QtWarningMsg:
        print('Qt gui warning:', msg)
    elif msgtype == QtMsgType.QtCriticalMsg:
        print('Qt gui critical error:', msg)
    elif msgtype == QtMsgType.QtFatalMsg:
        print('Qt gui fatal error:', msg)
    elif msgtype == QtMsgType.QtInfoMsg:
        print('Qt information message:', msg)
    elif msgtype == QtMsgType.QtDebugMsg:
        print('Qt debug message:', msg)


def start(hostname=None):
    # Install message handler for Qt messages
    qInstallMessageHandler(gui_msg_handler)

    # Start the Qt main object
    app = QApplication([])

    # Explicitly set font to avoid font loading warning dialogs
    app.setFont(QFont('Sans'))

    # Start the bluesky network client
    client = GuiClient()

    # Enable HiDPI support (Qt5 only)
    if QT_VERSION_STR[0] == '5' and QT_VERSION >= 0x050000:
        app.setAttribute(Qt.AA_UseHighDpiPixmaps)

    splash = Splash()

    # Register our custom pan/zoom event
    for etype in range(1000, 1000 + NUMCUSTOMEVENTS):
        reg_etype = QEvent.registerEventType(etype)
        if reg_etype != etype:
            print(('Warning: Registered event type differs from requested type id (%d != %d)' % (reg_etype, etype)))

    splash.show()

    # Install error message handler
    # handler = QErrorMessage.qtHandler()
    # handler.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)

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
        dialog = DiscoveryDialog(win)
        dialog.show()
        bs.net.start_discovery()

    else:
        client.connect(hostname=hostname)

    # Start the Qt main loop
    # app.exec_()
    app.exec()