""" QTGL Gui for BlueSky."""
try:
    from PyQt5.QtCore import Qt, QEvent, QT_VERSION, QT_VERSION_STR
    from PyQt5.QtWidgets import QApplication, QErrorMessage
    from PyQt5.QtOpenGL import QGLFormat

except ImportError:
    from PyQt4.QtCore import Qt, QEvent, QT_VERSION, QT_VERSION_STR
    from PyQt4.QtGui import QApplication, QErrorMessage
    from PyQt4.QtOpenGL import QGLFormat

# Local imports
from bluesky import settings
from bluesky.ui.qtgl.guiclient import GuiClient
from bluesky.ui.qtgl.mainwindow import MainWindow, Splash
from bluesky.ui.qtgl.customevents import NUMCUSTOMEVENTS


print(('Using Qt ' + QT_VERSION_STR + ' for windows and widgets'))

# Register settings defaults
settings.set_variable_defaults(scenario_path='scenario')

gui = None
client = GuiClient()

def init():
    global gui, client
    if gui is None:
        gui = Gui()
        splash = Splash()

        # Register our custom pan/zoom event
        for etype in range(1000, 1000 + NUMCUSTOMEVENTS):
            reg_etype = QEvent.registerEventType(etype)
            if reg_etype != etype:
                print(('Warning: Registered event type differs from requested type id (%d != %d)' % (reg_etype, etype)))

        splash.show()

        # Install error message handler
        handler = QErrorMessage.qtHandler()
        handler.setWindowFlags(Qt.WindowStaysOnTopHint)

        # Check and set OpenGL capabilities
        if not QGLFormat.hasOpenGL():
            raise RuntimeError('No OpenGL support detected for this system!')
        else:
            f = QGLFormat()
            f.setVersion(3, 3)
            f.setProfile(QGLFormat.CoreProfile)
            f.setDoubleBuffer(True)
            QGLFormat.setDefaultFormat(f)
            print(('QGLWidget initialized for OpenGL version %d.%d' % (f.majorVersion(), f.minorVersion())))


        splash.showMessage('Constructing main window')
        gui.processEvents()
        gui.init()
        client.connect(event_port=9000, stream_port=9001)
        splash.showMessage('Done!')
        gui.processEvents()
        splash.finish(gui.win)


def start():
    init()
    gui.start()


class Gui(QApplication):
    def __init__(self):
        super(Gui, self).__init__([])
        self.win = None
        self.initialized = False

        # Enable HiDPI support (Qt5 only)
        if QT_VERSION >= 0x050000:
            self.setAttribute(Qt.AA_UseHighDpiPixmaps)

    def init(self):
        ''' Create the main window and related widgets. '''
        self.win = MainWindow(self)
        self.win.show()
        self.initialized = True

    def start(self):
        self.exec_()

    def quit(self):
        # Send quit to server
        client.send_event(b'QUIT')
        self.closeAllWindows()

    def notify(self, receiver, event):
        # Send all key presses directly to the main window
        if event.type() == QEvent.KeyPress:
            return self.win.keyPressEvent(event)

        return super(Gui, self).notify(receiver, event)

    def __del__(self):
        # Make sure to Clean up at quit event
        self.aboutToQuit.connect(self.win.cleanUp)
