try:
    from PyQt4.QtOpenGL import QGLWidget
    QT_VERSION = 4
except ImportError:
    from PyQt5.QtOpenGL import QGLWidget
    QT_VERSION = 5

import OpenGL.GL as gl
from math import sin, cos, radians
import numpy as np

from glhelpers import BlueSkyProgram, RenderObject, TextObject, \
                      VERTEX_IS_LATLON, VERTEX_IS_METERS, VERTEX_IS_SCREEN, VERTEX_IS_GLXY


class ND(QGLWidget):
    def __init__(self, parent=None, shareWidget=None):
        super(ND, self).__init__(parent=parent, shareWidget=shareWidget)

        # Set size
        self.resize(400, 400)

    def create_objects(self):
        self.map = RenderObject(gl.GL_LINES)
        mapvertices = []
        for i in range(1, 5):
            for angle in range(-60, 60, max(2, 6 - 2 * i)):
                mapvertices.append(float(i) * 0.35 * sin(radians(angle)))
                mapvertices.append(-0.7 + float(i) * 0.35 * cos(radians(angle)))
                if i == 4:
                    mapvertices.append(float(i) * 0.35 * sin(radians(angle + 2)))
                    mapvertices.append(-0.7 + float(i) * 0.35 * cos(radians(angle + 2)))

        self.map.bind_vertex_attribute(np.array(mapvertices, dtype=np.float32))
        self.map.bind_color_attribute(np.array((1.0, 1.0, 1.0), dtype=np.float32))
        self.map.set_vertex_count(len(mapvertices))

        self.ticks = RenderObject(gl.GL_LINES, vertex_count=144)
        ticks = np.zeros(288, dtype=np.float32)
        for i in range(72):
            ticktop = 1.46 if i % 6 == 0 else (1.44 if i % 2 == 0 else 1.42)
            ticks[4*i  :4*i+2] = (1.4 * sin(radians(i * 5)), -0.7 + 1.4 * cos(radians(i * 5)))
            ticks[4*i+2:4*i+4] = (ticktop * sin(radians(i * 5)), -0.7 + ticktop * cos(radians(i * 5)))
        self.ticks.bind_vertex_attribute(ticks)
        self.ticks.bind_color_attribute(np.array((1.0, 1.0, 1.0), dtype=np.float32))

        self.ticklbls = TextObject(vertex_count=12 * 36)
        #self.ticklbls = RenderObject(gl.GL_TRIANGLES, vertex_count=12 * 36)
        ticklbls = np.zeros(24 * 36, dtype=np.float32)
        texcoords = np.zeros(36 * 36, dtype=np.float32)

        for i in range(36):
            if i % 3 == 0:
                w, h, y = 0.04, 0.08, 1.48
            else:
                w, h, y = 0.03, 0.06, 1.46
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
                ticklbls[24*i+2*j+1] -= 0.7

        self.ticklbls.bind_vertex_attribute(ticklbls)
        self.ticklbls.bind_texcoords_attribute(texcoords, size=3)
        self.ticklbls.bind_color_attribute(np.array((1.0, 1.0, 1.0), dtype=np.float32))

        self.ownship = RenderObject(gl.GL_LINES, vertex_count=6)
        self.ownship.bind_vertex_attribute(np.array([0.0, -0.7, 0.0, -0.82, 0.065, -0.73, -0.065, -0.73, 0.022, -0.8, -0.022, -0.8], dtype=np.float32))
        self.ownship.bind_color_attribute(np.array((1.0, 1.0, 0.0), dtype=np.float32))

        # Unbind VAO, VBO
        RenderObject.unbind_all()

        print 'is sharing? %d' % self.context().isSharing()

    def initializeGL(self):
        """Initialize OpenGL, VBOs, upload data on the GPU, etc."""

        # background color
        gl.glClearColor(0, 0, 0, 0)

        # Compile shaders and link color shader program
        self.color = BlueSkyProgram('data/graphics/shaders/shader.vert', 'data/graphics/shaders/color_shader.frag')

        # Compile shaders and link text shader program
        self.text = BlueSkyProgram('data/graphics/shaders/shader_text.vert', 'data/graphics/shaders/shader_text.frag')
        TextObject.init_shader(self.text)

        gl.glBindBuffer(gl.GL_UNIFORM_BUFFER, BlueSkyProgram.ubo_globaldata)
        gl.glBindBufferBase(gl.GL_UNIFORM_BUFFER, 1, BlueSkyProgram.ubo_globaldata)
        self.create_objects()

    def resizeGL(self, width, height):
        # paint within the whole window
        gl.glViewport(0, 0, width, height)

    def paintGL(self):
        """Paint the scene."""
        # pass if the framebuffer isn't complete yet
        if not gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) == gl.GL_FRAMEBUFFER_COMPLETE:
            return

        # clear the buffer
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        BlueSkyProgram.set_vertex_scale_type(VERTEX_IS_GLXY)

        # Select the non-textured shader
        self.color.use()
        self.map.draw()
        self.ticks.draw()
        self.ownship.draw()

        # Select the text shader
        self.text.use()
        self.ticklbls.draw()

        # Unbind everything
        RenderObject.unbind_all()
        gl.glUseProgram(0)
