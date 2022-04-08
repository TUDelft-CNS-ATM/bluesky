""" QTGL Gui for BlueSky."""
from bluesky.ui.qtgl.customevents import NUMCUSTOMEVENTS
from bluesky.ui.qtgl.guiclient import GuiClient
from bluesky.ui.qtgl.mainwindow import MainWindow, Splash, DiscoveryDialog

from PyQt5.QtCore import Qt, QEvent, qInstallMessageHandler, \
    QtWarningMsg, QtCriticalMsg, QtFatalMsg, \
    QT_VERSION, QT_VERSION_STR
from PyQt5.QtWidgets import QApplication, QErrorMessage

import bluesky as bs


print( ( 'Using Qt ' + QT_VERSION_STR + ' for windows and widgets' ) )


def gui_msg_handler( msgtype, context, msg ):
    if msgtype == QtWarningMsg:
        print( 'Qt gui warning:', msg )
    elif msgtype == QtCriticalMsg:
        print( 'Qt gui critical error:', msg )
    if msgtype == QtFatalMsg:
        print( 'Qt gui fatal error:', msg )
        exit()


def start(mode, hostname=None):
    # Install message handler for Qt messages
    qInstallMessageHandler( gui_msg_handler )

    # Start the Qt main object
    app = QApplication( [] )

    # Start the bluesky network client
    client = GuiClient()

    # Enable HiDPI support (Qt5 only)
    if QT_VERSION >= 0x050000:
        app.setAttribute( Qt.AA_UseHighDpiPixmaps )

    splash = Splash()

    # Register our custom pan/zoom event
    for etype in range( 1000, 1000 + NUMCUSTOMEVENTS ):
        reg_etype = QEvent.registerEventType( etype )
        if reg_etype != etype:
            print( ( 'Warning: Registered event type differs from requested type id (%d != %d)' % ( reg_etype, etype ) ) )

    splash.show()

    # Install error message handler
    handler = QErrorMessage.qtHandler()
    handler.setWindowFlags( Qt.WindowStaysOnTopHint )

    splash.showMessage( 'Constructing main window' )
    app.processEvents()
    win = MainWindow( mode )
    win.show()
    splash.showMessage( 'Done!' )
    app.processEvents()
    splash.finish( win )
    # If this instance of the gui is started in client-only mode, show
    # server selection dialog
    if mode == 'client' and hostname is None:
        dialog = DiscoveryDialog(win)
        dialog.show()
        bs.net.start_discovery()

    else:
        client.connect(hostname=hostname or 'localhost',
                       event_port=bs.settings.event_port,
                       stream_port=bs.settings.stream_port)

    # Start the Qt main loop
    app.exec_()
