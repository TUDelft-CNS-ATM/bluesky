""" QTGL Gui for BlueSky."""
try:
    from PyQt5.QtCore import Qt, QEvent, QT_VERSION, QT_VERSION_STR
    from PyQt5.QtWidgets import QApplication

except ImportError:
    from PyQt4.QtCore import Qt, QEvent, QT_VERSION, QT_VERSION_STR
    from PyQt4.QtGui import QApplication

# Local imports
from bluesky import settings

from bluesky.ui.qtgl.mainwindow import MainWindow

from . import guiio as io

print(('Using Qt ' + QT_VERSION_STR + ' for windows and widgets'))

# Register settings defaults
settings.set_variable_defaults(scenario_path='scenario')

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
        io.send_event(b'QUIT')
        self.closeAllWindows()

    def notify(self, receiver, event):
        # Send all key presses directly to the main window
        if event.type() == QEvent.KeyPress:
            return self.win.keyPressEvent(event)

        return super(Gui, self).notify(receiver, event)

    def __del__(self):
        # Make sure to Clean up at quit event
        self.aboutToQuit.connect(self.win.cleanUp)
