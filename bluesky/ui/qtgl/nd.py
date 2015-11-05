try:
    from PyQt4.QtCore import qCritical, QTimer
    from PyQt4.QtOpenGL import QGLWidget
    QT_VERSION = 4
except ImportError:
    from PyQt5.QtCore import qCritical, QTimer
    from PyQt5.QtOpenGL import QGLWidget
    QT_VERSION = 5

import OpenGL.GL as gl
from math import sin, cos, radians
import numpy as np
from ctypes import c_float, c_int, Structure

from glhelpers import BlueSkyProgram, RenderObject, TextObject, UniformBuffer

VERTEX_IS_LATLON, VERTEX_IS_METERS, VERTEX_IS_SCREEN, VERTEX_IS_GLXY = range(4)
ATTRIB_VERTEX, ATTRIB_TEXCOORDS, ATTRIB_LAT, ATTRIB_LON, ATTRIB_ORIENTATION, ATTRIB_COLOR, ATTRIB_TEXDEPTH = range(7)

lightblue2           = (0.58, 0.7, 0.92)
lightblue3           = (0.86, 0.98, 1.0)


class ndUBO(UniformBuffer):
    class Data(Structure):
        _fields_ = [("ownhdg", c_float), ("ownlat", c_float), ("ownlon", c_float),
        ("zoom", c_float), ("ownid", c_int), ("vertex_modifiers", c_int)]

    data = Data(0.0, 0.0, 0.0, 4.0, 3)

    def __init__(self):
        super(ndUBO, self).__init__(self.data)

    def set_zoom(self, zoom):
        self.data.zoom   = zoom

    def set_owndata(self, acid, lat, lon, hdg):
        self.data.ownid  = acid
        self.data.ownlat = lat
        self.data.ownlon = lon
        self.data.ownhdg = hdg

    def set_vertex_modifiers(self, scale_type, rotate_ownhdg):
        self.data.vertex_modifiers = scale_type + (10 if rotate_ownhdg else 0)
        self.update()


class ND(QGLWidget):
    def __init__(self, parent=None, shareWidget=None):
        super(ND, self).__init__(parent=parent, shareWidget=shareWidget)

        self.shareWidget = shareWidget
        self.ac_id = ''
        self.n_aircraft = None
        self.initialized = False
        self.invalid_count = 0
        # Set size
        self.viewport = (0, 0, 400, 400)
        self.resize(400, 400)

    def setAircraftID(self, ac_id):
        self.ac_id = ac_id
        self.setWindowTitle(ac_id)

    def update_aircraft_data(self, ownid, ownlat, ownlon, owntas, ownhdg, n_aircraft):
        self.globaldata.set_owndata(ownid, ownlat, ownlon, ownhdg)
        self.n_aircraft = n_aircraft

    def create_objects(self):
        if not self.isValid():
            self.invalid_count += 1
            print 'ND: Context not valid in create_objects, count=%d' % self.invalid_count
            QTimer.singleShot(100, self.create_objects)
            return

        # Make the nd widget context current, necessary when create_objects is not called from initializeGL
        self.makeCurrent()

        self.edge = RenderObject(gl.GL_LINE_STRIP, vertex_count=60)
        edge = np.zeros(120, dtype=np.float32)
        edge[0:120:2] = 1.4 * np.sin(np.radians(np.arange(-60, 60, 2)))
        edge[1:120:2] = 1.4 * np.cos(np.radians(np.arange(-60, 60, 2)))
        self.edge.bind_attrib(ATTRIB_VERTEX, 2, edge)
        self.edge.bind_attrib(ATTRIB_COLOR, 3, np.array((1.0, 1.0, 1.0), dtype=np.float32), instance_divisor=1)
        self.arcs = RenderObject(gl.GL_LINES)
        arcs = []
        for i in range(1, 4):
            for angle in range(-60, 60, max(2, 6 - 2 * i)):
                arcs.append(float(i) * 0.35 * sin(radians(angle)))
                arcs.append(float(i) * 0.35 * cos(radians(angle)))
                if i == 4:
                    arcs.append(float(i) * 0.35 * sin(radians(angle + 2)))
                    arcs.append(float(i) * 0.35 * cos(radians(angle + 2)))

        self.arcs.bind_attrib(ATTRIB_VERTEX, 2, np.array(arcs, dtype=np.float32))
        self.arcs.bind_attrib(ATTRIB_COLOR, 3, np.array((1.0, 1.0, 1.0), dtype=np.float32), instance_divisor=1)
        self.arcs.set_vertex_count(len(arcs))

        self.mask = RenderObject(gl.GL_TRIANGLE_STRIP, vertex_count=120)
        mask = []
        for angle in range(-60, 60, 2):
            mask.append(1.4*sin(radians(angle)))
            mask.append(10.0)
            mask.append(1.4*sin(radians(angle)))
            mask.append(1.4*cos(radians(angle)))
        self.mask.bind_attrib(ATTRIB_VERTEX, 2, np.array(mask, dtype=np.float32))
        self.mask.bind_attrib(ATTRIB_COLOR, 3, np.array((0.0, 0.0, 0.0), dtype=np.float32), instance_divisor=1)

        self.ticks = RenderObject(gl.GL_LINES, vertex_count=144)
        ticks = np.zeros(288, dtype=np.float32)
        for i in range(72):
            ticktop = 1.46 if i % 6 == 0 else (1.44 if i % 2 == 0 else 1.42)
            ticks[4*i  :4*i+2] = (1.4 * sin(radians(i * 5)), 1.4 * cos(radians(i * 5)))
            ticks[4*i+2:4*i+4] = (ticktop * sin(radians(i * 5)), ticktop * cos(radians(i * 5)))
        self.ticks.bind_attrib(ATTRIB_VERTEX, 2, ticks)
        self.ticks.bind_attrib(ATTRIB_COLOR, 3, np.array((1.0, 1.0, 1.0), dtype=np.float32), instance_divisor=1)

        self.ticklbls = TextObject(vertex_count=12 * 36)
        #self.ticklbls = RenderObject(gl.GL_TRIANGLES, vertex_count=12 * 36)
        ticklbls = np.zeros(24 * 36, dtype=np.float32)
        texcoords = np.zeros(36 * 36, dtype=np.float32)

        for i in range(36):
            if i % 3 == 0:
                w, h, y = 0.045, 0.09, 1.48
            else:
                w, h, y = 0.035, 0.07, 1.46
            tmp = [(-w, h+y), (-w, y), (0.0, h+y), (0.0, h+y), (-w, y), (0.0, y),
                   (0.0, h+y), (0.0, y), (w, h+y), (w, h+y), (0.0, y), (w, y)]

            # numerics start at ASCII 48
            c1 = i / 10 + 48
            c2 = i % 10 + 48
            texcoords[36*i:36*i+18]    = [0, 0, c1, 0, 1, c1, 1, 0, c1, 1, 0, c1, 0, 1, c1, 1, 1, c1]
            texcoords[36*i+18:36*i+36] = [0, 0, c2, 0, 1, c2, 1, 0, c2, 1, 0, c2, 0, 1, c2, 1, 1, c2]
            angle = radians(10 * (36 - i))
            rot = np.array([[cos(angle), -sin(angle)], [sin(angle), cos(angle)]])
            for j in range(12):
                ticklbls[24*i+2*j:24*i+2*j+2] = rot.dot(tmp[j])

        self.ticklbls.bind_attrib(ATTRIB_VERTEX, 2, ticklbls)
        self.ticklbls.bind_attrib(ATTRIB_TEXCOORDS, 3, texcoords)
        self.ticklbls.bind_attrib(ATTRIB_COLOR, 3, np.array((1.0, 1.0, 1.0), dtype=np.float32), instance_divisor=1)

        self.ownship = RenderObject(gl.GL_LINES, vertex_count=6)
        self.ownship.bind_attrib(ATTRIB_VERTEX, 2, np.array([0.0, 0.0, 0.0, -0.12, 0.065, -0.03, -0.065, -0.03, 0.022, -0.1, -0.022, -0.1], dtype=np.float32))
        self.ownship.bind_attrib(ATTRIB_COLOR, 3, np.array((1.0, 1.0, 0.0), dtype=np.float32), instance_divisor=1)

        self.spdlabel_text  = TextObject()
        self.spdlabel_text.prepare_text_string('GS    TAS', 0.05, (1.0, 1.0, 1.0), (-0.98, 1.6))
        self.spdlabel_val = TextObject()
        self.spdlabel_val.prepare_text_string('  000    000', 0.05, (0.0, 1.0, 0.0), (-0.97, 1.6))

        self.waypoints = RenderObject.copy(self.shareWidget.waypoints)
        self.wptlabels = RenderObject.copy(self.shareWidget.wptlabels)
        self.airports  = RenderObject.copy(self.shareWidget.airports)
        self.aptlabels = RenderObject.copy(self.shareWidget.aptlabels)
        self.protectedzone = RenderObject.copy(self.shareWidget.protectedzone)
        self.ac_symbol = RenderObject.copy(self.shareWidget.ac_symbol)
        self.aclabels = RenderObject.copy(self.shareWidget.aclabels)

        # Unbind VAO, VBO
        RenderObject.unbind_all()

        # Done initializing
        self.initialized = True

    def initializeGL(self):
        """Initialize OpenGL, VBOs, upload data on the GPU, etc."""

        # First check for supported GL version
        gl_version = float(gl.glGetString(gl.GL_VERSION)[:3])
        if gl_version < 3.3:
            return

        # background color
        gl.glClearColor(0, 0, 0, 0)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        self.globaldata = ndUBO()

        try:
            # Compile shaders and link color shader program
            self.color = BlueSkyProgram('data/graphics/shaders/nd-normal.vert', 'data/graphics/shaders/nd-color.frag')
            self.color.bind_uniform_buffer('global_data', self.globaldata)

            # Compile shaders and link text shader program
            self.text = BlueSkyProgram('data/graphics/shaders/nd-text.vert', 'data/graphics/shaders/nd-text.frag')
            self.text.bind_uniform_buffer('global_data', self.globaldata)
            TextObject.init_shader(self.text)

        except RuntimeError as e:
            qCritical('Error compiling shaders in radarwidget: ' + e.args[0])
            return

        # Set initial zoom
        self.globaldata.set_zoom(4.0)

        self.create_objects()

    def resizeGL(self, width, height):
        # paint within the largest possible rectangular area in the window
        w = h = min(width, height)
        x = max(0, (width - w) / 2)
        y = max(0, (height - h) / 2)
        self.viewport = (x, y, w, h)

    def paintGL(self):
        """Paint the scene."""
        # pass if the framebuffer isn't complete yet or if not initialized
        if not (gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) == gl.GL_FRAMEBUFFER_COMPLETE and self.initialized and self.isVisible()):
            return

        # Set the viewport and clear the framebuffer
        gl.glViewport(*self.viewport)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        # Select the non-textured shader
        self.color.use()

        self.globaldata.set_vertex_modifiers(VERTEX_IS_GLXY, False)
        self.arcs.draw()

        self.globaldata.set_vertex_modifiers(VERTEX_IS_METERS, False)
        self.protectedzone.draw(n_instances=self.n_aircraft)
        self.globaldata.set_vertex_modifiers(VERTEX_IS_SCREEN, True)
        self.ac_symbol.draw(n_instances=self.n_aircraft)

        self.globaldata.set_vertex_modifiers(VERTEX_IS_SCREEN, False)
        self.waypoints.bind()
        gl.glVertexAttrib3f(ATTRIB_COLOR, *lightblue2)
        self.waypoints.draw()
        self.airports.bind()
        gl.glVertexAttrib3f(ATTRIB_COLOR, *lightblue2)
        self.airports.draw()

        self.text.use()
        self.wptlabels.bind()
        gl.glVertexAttrib3f(ATTRIB_COLOR, *lightblue3)
        self.wptlabels.draw(n_instances=self.waypoints.n_instances)
        self.aptlabels.bind()
        gl.glVertexAttrib3f(ATTRIB_COLOR, *lightblue3)
        self.aptlabels.draw(n_instances=self.airports.n_instances)
        self.aclabels.draw(n_instances=self.n_aircraft)

        self.color.use()
        self.globaldata.set_vertex_modifiers(VERTEX_IS_GLXY, False)
        self.ownship.draw()
        self.mask.draw()
        self.edge.draw()

        self.globaldata.set_vertex_modifiers(VERTEX_IS_GLXY, True)
        self.ticks.draw()

        # Select the text shader
        self.text.use()
        self.ticklbls.draw()

        self.globaldata.set_vertex_modifiers(VERTEX_IS_GLXY, False)
        self.spdlabel_text.draw()
        self.spdlabel_val.draw()

        # Unbind everything
        RenderObject.unbind_all()
        gl.glUseProgram(0)
