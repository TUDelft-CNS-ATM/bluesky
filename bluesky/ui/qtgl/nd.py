""" Navigation display for the QTGL gui."""
from PyQt5.QtCore import qCritical, QTimer
from PyQt5.QtOpenGL import QGLWidget
import OpenGL.GL as gl
from math import sin, cos, radians
import numpy as np
from ctypes import c_float, c_int, Structure

import bluesky as bs
from .glhelpers import BlueSkyProgram, RenderObject, UniformBuffer

VERTEX_IS_LATLON, VERTEX_IS_METERS, VERTEX_IS_SCREEN, VERTEX_IS_GLXY = list(range(4))
ATTRIB_VERTEX, ATTRIB_TEXCOORDS, ATTRIB_LAT, ATTRIB_LON, ATTRIB_ORIENTATION, ATTRIB_COLOR, ATTRIB_TEXDEPTH = list(range(7))

white                = (255, 255, 255)
black                = (0,   0,   0)
yellow               = (255, 255, 0)
green                = (0,   255, 0)
lightblue2           = (148, 178, 235)
lightblue3           = (220, 250, 255)


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

        # Connect to incoming stream data
        bs.net.stream_received.connect(self.on_simstream_received)

    def on_simstream_received(self, streamname, data, sender_id):
        if streamname == b'ACDATA':
            if self.ac_id in data['id']:
                idx = data['id'].index(self.ac_id.upper())
                lat = data['lat'][idx]
                lon = data['lon'][idx]
                trk = data['trk'][idx]
                tas = data['tas'][idx]
                self.n_aircraft = len(data['lat'])
                self.globaldata.set_owndata(idx, lat, lon, trk)

    def setAircraftID(self, ac_id):
        self.ac_id = ac_id
        self.setWindowTitle(ac_id)

    def create_objects(self):
        if not (self.isValid() and self.shareWidget.initialized):
            self.invalid_count += 1
            print('ND: Context not valid in create_objects, or shareWidget not yet initialized')
            QTimer.singleShot(100, self.create_objects)
            return

        # Make the nd widget context current, necessary when create_objects is not called from initializeGL
        self.makeCurrent()

        # Use the same font as the radarwidget
        self.font = self.shareWidget.font.copy()
        self.font.init_shader(self.text_shader)

        edge = np.zeros(120, dtype=np.float32)
        edge[0:120:2] = 1.4 * np.sin(np.radians(np.arange(-60, 60, 2)))
        edge[1:120:2] = 1.4 * np.cos(np.radians(np.arange(-60, 60, 2)))
        self.edge = RenderObject(gl.GL_LINE_STRIP, vertex=edge, color=white)

        arcs = []
        for i in range(1, 4):
            for angle in range(-60, 60, max(2, 6 - 2 * i)):
                arcs.append(float(i) * 0.35 * sin(radians(angle)))
                arcs.append(float(i) * 0.35 * cos(radians(angle)))
                if i == 4:
                    arcs.append(float(i) * 0.35 * sin(radians(angle + 2)))
                    arcs.append(float(i) * 0.35 * cos(radians(angle + 2)))
        arcs = np.array(arcs, dtype=np.float32)
        self.arcs = RenderObject(gl.GL_LINES, vertex=arcs, color=white)

        mask = []
        for angle in range(-60, 60, 2):
            mask.append(1.4 * sin(radians(angle)))
            mask.append(10.0)
            mask.append(1.4 * sin(radians(angle)))
            mask.append(1.4 * cos(radians(angle)))
        mask = np.array(mask, dtype=np.float32)
        self.mask = RenderObject(gl.GL_TRIANGLE_STRIP, vertex=mask, color=black)

        ticks = np.zeros(288, dtype=np.float32)
        for i in range(72):
            ticktop = 1.46 if i % 6 == 0 else (1.44 if i % 2 == 0 else 1.42)
            ticks[4*i  :4*i+2] = (1.4 * sin(radians(i * 5)), 1.4 * cos(radians(i * 5)))
            ticks[4*i+2:4*i+4] = (ticktop * sin(radians(i * 5)), ticktop * cos(radians(i * 5)))
        self.ticks = RenderObject(gl.GL_LINES, vertex=ticks, color=white)

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

        self.ticklbls = RenderObject(gl.GL_TRIANGLES, vertex=ticklbls, color=white, texcoords=texcoords)

        vown = np.array([0.0, 0.0, 0.0, -0.12, 0.065, -0.03, -0.065, -0.03, 0.022, -0.1, -0.022, -0.1], dtype=np.float32)
        self.ownship = RenderObject(gl.GL_LINES, vertex=vown, color=yellow)

        self.spdlabel_text = self.font.prepare_text_string('GS    TAS', 0.05, white, (-0.98, 1.6))
        self.spdlabel_val  = self.font.prepare_text_string('  000    000', 0.05, green, (-0.97, 1.6))

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
            self.color_shader = BlueSkyProgram('data/graphics/shaders/nd-normal.vert', 'data/graphics/shaders/nd-color.frag')
            self.color_shader.bind_uniform_buffer('global_data', self.globaldata)

            # Compile shaders and link text shader program
            self.text_shader = BlueSkyProgram('data/graphics/shaders/nd-text.vert', 'data/graphics/shaders/nd-text.frag')
            self.text_shader.bind_uniform_buffer('global_data', self.globaldata)

        except RuntimeError as e:
            qCritical('Error compiling shaders in radarwidget: ' + e.args[0])
            return

        # Set initial zoom
        self.globaldata.set_zoom(4.0)

        self.create_objects()

    def resizeGL(self, width, height):
        # paint within the largest possible rectangular area in the window
        w = h = min(width, height)
        x = max(0, (width - w) // 2)
        y = max(0, (height - h) // 2)
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
        self.color_shader.use()

        self.globaldata.set_vertex_modifiers(VERTEX_IS_GLXY, False)
        self.arcs.draw()

        self.globaldata.set_vertex_modifiers(VERTEX_IS_METERS, False)
        self.protectedzone.draw(n_instances=self.n_aircraft)
        self.globaldata.set_vertex_modifiers(VERTEX_IS_SCREEN, True)
        self.ac_symbol.draw(n_instances=self.n_aircraft)

        self.globaldata.set_vertex_modifiers(VERTEX_IS_SCREEN, False)
        self.waypoints.bind()
        gl.glVertexAttrib4Nub(ATTRIB_COLOR, *(lightblue2 + (255,)))
        self.waypoints.draw()
        self.airports.bind()
        gl.glVertexAttrib4Nub(ATTRIB_COLOR, *(lightblue2 + (255,)))
        self.airports.draw()

        self.text_shader.use()
        self.font.use()
        self.font.set_char_size(self.wptlabels.char_size)
        self.font.set_block_size(self.wptlabels.block_size)
        self.wptlabels.bind()
        gl.glVertexAttrib4Nub(ATTRIB_COLOR, *(lightblue3 + (255,)))
        self.wptlabels.draw(n_instances=self.waypoints.n_instances)

        self.font.set_char_size(self.aptlabels.char_size)
        self.font.set_block_size(self.aptlabels.block_size)
        self.aptlabels.bind()
        gl.glVertexAttrib4Nub(ATTRIB_COLOR, *(lightblue3 + (255,)))
        self.aptlabels.draw(n_instances=self.airports.n_instances)

        self.font.set_char_size(self.aclabels.char_size)
        self.font.set_block_size(self.aclabels.block_size)
        self.aclabels.draw(n_instances=self.n_aircraft)

        self.color_shader.use()
        self.globaldata.set_vertex_modifiers(VERTEX_IS_GLXY, False)
        self.ownship.draw()
        self.mask.draw()
        self.edge.draw()

        self.globaldata.set_vertex_modifiers(VERTEX_IS_GLXY, True)
        self.ticks.draw()

        # Select the text shader
        self.text_shader.use()
        self.font.use()
        self.font.set_block_size((0, 0))
        self.ticklbls.draw()

        self.globaldata.set_vertex_modifiers(VERTEX_IS_GLXY, False)
        self.spdlabel_text.draw()
        self.spdlabel_val.draw()

        # Unbind everything
        RenderObject.unbind_all()
        gl.glUseProgram(0)
