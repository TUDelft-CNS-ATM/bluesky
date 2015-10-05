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
    def __init__(self, shareWidget=None):
        super(ND, self).__init__(shareWidget=shareWidget)

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

        self.ownship = RenderObject(gl.GL_LINES, vertex_count=6)
        self.ownship.bind_vertex_attribute(np.array([0.0, -0.7, 0.0, -0.82, 0.065, -0.73, -0.065, -0.73, 0.022, -0.8, -0.022, -0.8], dtype=np.float32))
        self.ownship.bind_color_attribute(np.array((1.0, 1.0, 0.0), dtype=np.float32))

        # Unbind VAO, VBO
        RenderObject.unbind_all()

    def initializeGL(self):
        """Initialize OpenGL, VBOs, upload data on the GPU, etc."""

        # background color
        gl.glClearColor(0, 0, 0, 0)

        # Compile shaders and link color shader program
        self.color = BlueSkyProgram('data/graphics/shaders/shader.vert', 'data/graphics/shaders/color_shader.frag')

        # Compile shaders and link text shader program
        self.text = BlueSkyProgram('data/graphics/shaders/shader_text.vert', 'data/graphics/shaders/shader_text.frag')
        TextObject.init_shader(self.text)

        self.create_objects()

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
        self.ownship.draw()

        # Unbind everything
        RenderObject.unbind_all()
        gl.glUseProgram(0)
