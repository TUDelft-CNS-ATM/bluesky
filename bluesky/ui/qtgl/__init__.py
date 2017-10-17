''' BlueSky QtGL graphical interface module. '''
try:
    from PyQt5.QtCore import Qt, QEvent
    from PyQt5.QtWidgets import QErrorMessage
    from PyQt5.QtOpenGL import QGLFormat

except ImportError:
    from PyQt4.QtCore import Qt, QEvent
    from PyQt4.QtGui import QErrorMessage
    from PyQt4.QtOpenGL import QGLFormat

from .gui import Gui
from . import guiio
from .mainwindow import MainWindow, Splash
from .radarwidget import RadarWidget
from .glhelpers import BlueSkyProgram, RenderObject

from bluesky.simulation.qtgl import NUMEVENTS

gui = None

def init():
    global gui
    if gui is None:
        gui = Gui()
        splash = Splash()

        # Register our custom pan/zoom event
        for etype in range(1000, 1000 + NUMEVENTS):
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

        guiio.init()

        splash.showMessage('Constructing main window')
        gui.processEvents()
        gui.init()
        splash.showMessage('Done!')
        gui.processEvents()
        splash.finish(gui.win)


def start():
    init()
    gui.start()
