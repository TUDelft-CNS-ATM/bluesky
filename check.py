#!/usr/bin/python
from __future__ import print_function
import traceback

import PyQt6
print("This script checks the availability of the libraries required by BlueSky, and the capabilities of your system.")
print()
np = sp = mpl = qt = gl = glhw = pg = False

# Basic libraries
print("Checking for numpy              ", end=' ')
try:
    import numpy
except ImportError:
    print("[FAIL]")
else:
    np = True
    print('[OK]')

print("Checking for scipy              ", end=' ')
try:
    import scipy
except ImportError:
    print("[FAIL]")
else:
    sp = True
    print('[OK]')

print("Checking for matplotlib         ", end=' ')
try:
    import matplotlib
except ImportError:
    print("[FAIL]")
else:
    mpl = True
    print('[OK]')

# Graphical libs and capabilities

print("Checking for pyqt               ", end=' ')
try:
    from PyQt6.QtCore import QT_VERSION_STR
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtGui import QSurfaceFormat as QGLFormat
    from PyQt6.QtOpenGLWidgets import QOpenGLWidget as QGLWidget
    
    qt = True
except ImportError:
    print("[FAIL]")

if qt:
    print(f"[QT {QT_VERSION_STR}]")
    print("Checking for pyopengl           ", end=' ')
    try:
        import OpenGL
        import OpenGL.GL as ogl
    except ImportError:
        print("[FAIL]")
    else:
        v = OpenGL.__version__.split('.')
        ver = float(v[0]) + 0.1 * int(v[1])

        gl = (ver >= 3.1)
        if gl:
            print('[OK]')
        else:
            print('[FAIL]')
            print('OpenGL module version should be at least 3.1.0')
        print('OpenGL module version is         [%s]' % OpenGL.__version__)
        print("Checking GL capabilities        ", end=' ')
        app = QApplication([])

        if not QGLFormat.hasOpenGL():
            print('[FAIL]')
        else:
            print('[OK]')
            print('GL Version at least 3.3         ', end=' ')
            try:
                f = QGLFormat()
                f.setVersion(3, 3)
                f.setProfile(QGLFormat.OpenGLContextProfile.CoreProfile)
                f.setDoubleBuffer(True)
                QGLFormat.setDefaultFormat(f)

                class GLTest(QGLWidget):
                    gl_version = 0.0
                    def initializeGL(self):
                        GLTest.gl_version = float(ogl.glGetString(ogl.GL_VERSION)[:3])

                test = GLTest()

                test.show()

                if GLTest.gl_version >= 3.3:
                    print("[OK]")
                    glhw = True
                else:
                    print("[FAIL]")

                print("Supported GL version             [%.1f]" % GLTest.gl_version)
            except:
                print('[FAIL]')
                print('Could not determine GL version')

            print("Checking for pyopengl-accelerate", end=' ')
        try:
            import OpenGL_accelerate
        except ImportError:
            print("[FAIL]")
        else:
            print('[OK]')


print("Checking for pygame             ", end=' ')
try:
    import pygame
except ImportError:
    print('[FAIL]')
else:
    pg = True
    print('[OK]')

print()
if np and sp and mpl:
    canrunqt = (qt and gl and glhw)
    canrunpg = pg
    if canrunpg or canrunqt:
        print('You have all the required libraries to run BlueSky. You can use', end=' ')
    if canrunpg and canrunqt:
        print('both the QTGL and the pygame versions.')
    else:
        print('only the %s version.' % ('pygame' if canrunpg else 'QTGL'))

print("Checking bluesky modules")
try:
    import bluesky
    bluesky.init(mode='client')
    from bluesky.ui import *
    from bluesky.stack import *
    from bluesky.simulation import *
    from bluesky.traffic import *
    from bluesky.traffic.asas import *
    from bluesky.traffic.performance import *
    from bluesky.navdatabase import *
except Exception as err:
    print(traceback.format_exc())
    print("One or more BlueSky modules are not working properly, check the above error for more detail.")
else:
    print("Successfully loaded all BlueSky modules. Start BlueSky by running BlueSky.py.")
