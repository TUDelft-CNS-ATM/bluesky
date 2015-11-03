try:
    from PyQt5.QtCore import QTimer
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtOpenGL import QGLWidget, QGLFormat, QGLContext
    QT_VERSION = 5
except:
    from PyQt4.QtCore import QTimer
    from PyQt4.QtGui import QApplication
    from PyQt4.QtOpenGL import QGLWidget, QGLFormat, QGLContext
    QT_VERSION = 4
import OpenGL.GL as gl


class GLTest(QGLWidget):
    def __init__(self, parent=None):
        self.first = True
        f = QGLFormat()
        f.setVersion(3, 3)
        f.setProfile(QGLFormat.CoreProfile)
        f.setDoubleBuffer(True)
        if QT_VERSION == 4:
            QGLWidget.__init__(self, QGLContext(f, None), parent)
        else:
            # Qt 5
            QGLWidget.__init__(self, QGLContext(f), parent)

        print('QGLWidget initialized for OpenGL version %d.%d' % (f.majorVersion(), f.minorVersion()))

        print("This script creates a file called 'opengl-test.txt', containing information about the opengl support of your computer. Useful when debugging problems with the opengl version of BlueSky.")

    def initializeGL(self):
        gl.glClearColor(0, 0, 0, 0)

    def paintGL(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        if not self.first:
            return
        f = open('opengl_test.txt', 'w')
        f.write('Supported OpenGL version: ' + gl.glGetString(gl.GL_VERSION) + '\n')
        f.write('Supported GLSL version: ' + gl.glGetString(gl.GL_SHADING_LANGUAGE_VERSION) + '\n')
        numext = gl.glGetIntegerv(gl.GL_NUM_EXTENSIONS)
        f.write('Supported OpenGL extensions:' + '\n')
        extensions = ''
        for i in range(numext):
            extensions += ', ' + gl.glGetStringi(gl.GL_EXTENSIONS, i)
        f.write(extensions)
        f.close()
        self.first = False

if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    test = GLTest()

    timer = QTimer(test)
    timer.timeout.connect(test.updateGL)
    timer.start(50)

    test.show()
    app.exec_()
