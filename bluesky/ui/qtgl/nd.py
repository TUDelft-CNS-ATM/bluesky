try:
    from PyQt4.QtOpenGL import QGLWidget
    QT_VERSION = 4
except ImportError:
    from PyQt5.QtOpenGL import QGLWidget
    QT_VERSION = 5

import OpenGL.GL as gl
from math import sin, cos, radians
import numpy as np
from ctypes import c_float, c_int, Structure

from glhelpers import BlueSkyProgram, RenderObject, TextObject, UniformBuffer


class ndUBO(UniformBuffer):
    class Data(Structure):
        _fields_ = [("ownhdg", c_float), ("ownlat", c_float), ("ownlon", c_float),
        ("zoom", c_float), ("screen_width", c_int), ("screen_height", c_int), ("vertex_scale_type", c_int)]

    data = Data()

    def __init__(self):
        super(ndUBO, self).__init__(self.data)

    def set_zoom(self, zoom):
        self.data.zoom   = zoom

    def set_pos_and_hdg(self, lat, lon, hdg):
        self.data.ownlat = lat
        self.data.ownlon = lon
        self.data.ownhdg = hdg

    def set_win_width_height(self, w, h):
        self.data.screen_width  = w
        self.data.screen_height = h

    def enable_hdg_rotate(self, flag=True):
        if not flag:
            ownhdg = self.data.ownhdg
            self.data.ownhdg = 0
            self.update(0, 4)
            self.data.ownhdg = ownhdg
        else:
            self.update(0, 4)


class ND(QGLWidget):
    def __init__(self, parent=None, shareWidget=None):
        super(ND, self).__init__(parent=parent, shareWidget=shareWidget)

        self.ac_id = ''

        # Set size
        self.resize(400, 400)

    def setAircraftID(self, ac_id):
        self.ac_id = ac_id
        self.setWindowTitle(ac_id)

    def update_aircraft_data(self, ownlat, ownlon, ownhdg):
        self.globaldata.set_pos_and_hdg(ownlat, ownlon, ownhdg)

    def create_objects(self):
        self.map = RenderObject(gl.GL_LINES)
        mapvertices = []
        for i in range(1, 5):
            for angle in range(-60, 60, max(2, 6 - 2 * i)):
                mapvertices.append(float(i) * 0.35 * sin(radians(angle)))
                mapvertices.append(float(i) * 0.35 * cos(radians(angle)))
                if i == 4:
                    mapvertices.append(float(i) * 0.35 * sin(radians(angle + 2)))
                    mapvertices.append(float(i) * 0.35 * cos(radians(angle + 2)))

        self.map.bind_vertex_attribute(np.array(mapvertices, dtype=np.float32))
        self.map.bind_color_attribute(np.array((1.0, 1.0, 1.0), dtype=np.float32))
        self.map.set_vertex_count(len(mapvertices))

        self.ticks = RenderObject(gl.GL_LINES, vertex_count=144)
        ticks = np.zeros(288, dtype=np.float32)
        for i in range(72):
            ticktop = 1.46 if i % 6 == 0 else (1.44 if i % 2 == 0 else 1.42)
            ticks[4*i  :4*i+2] = (1.4 * sin(radians(i * 5)), 1.4 * cos(radians(i * 5)))
            ticks[4*i+2:4*i+4] = (ticktop * sin(radians(i * 5)), ticktop * cos(radians(i * 5)))
        self.ticks.bind_vertex_attribute(ticks)
        self.ticks.bind_color_attribute(np.array((1.0, 1.0, 1.0), dtype=np.float32))

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

        self.ticklbls.bind_vertex_attribute(ticklbls)
        self.ticklbls.bind_texcoords_attribute(texcoords, size=3)
        self.ticklbls.bind_color_attribute(np.array((1.0, 1.0, 1.0), dtype=np.float32))

        self.ownship = RenderObject(gl.GL_LINES, vertex_count=6)
        self.ownship.bind_vertex_attribute(np.array([0.0, 0.0, 0.0, -0.12, 0.065, -0.03, -0.065, -0.03, 0.022, -0.1, -0.022, -0.1], dtype=np.float32))
        self.ownship.bind_color_attribute(np.array((1.0, 1.0, 0.0), dtype=np.float32))

        # Unbind VAO, VBO
        RenderObject.unbind_all()

    def initializeGL(self):
        """Initialize OpenGL, VBOs, upload data on the GPU, etc."""

        # background color
        gl.glClearColor(0, 0, 0, 0)

        self.globaldata = ndUBO()

        # Compile shaders and link color shader program
        self.color = BlueSkyProgram('data/graphics/shaders/nd-normal.vert', 'data/graphics/shaders/nd-color.frag')
        self.color.bind_uniform_buffer('global_data', self.globaldata)

        # Compile shaders and link text shader program
        self.text = BlueSkyProgram('data/graphics/shaders/nd-text.vert', 'data/graphics/shaders/nd-text.frag')
        self.text.bind_uniform_buffer('global_data', self.globaldata)
        TextObject.init_shader(self.text)

        self.create_objects()

    def resizeGL(self, width, height):
        # paint within the whole window
        gl.glViewport(0, 0, width, height)

    def paintGL(self):
        """Paint the scene."""
        # pass if the framebuffer isn't complete yet
        if not gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) == gl.GL_FRAMEBUFFER_COMPLETE:
            return

        # Update uniform global data
        self.globaldata.update()

        # clear the buffer
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        self.globaldata.enable_hdg_rotate(False)

        # Select the non-textured shader
        self.color.use()
        self.map.draw()
        self.ownship.draw()

        self.globaldata.enable_hdg_rotate(True)
        self.ticks.draw()

        # Select the text shader
        self.text.use()
        self.ticklbls.draw()

        # Unbind everything
        RenderObject.unbind_all()
        gl.glUseProgram(0)
