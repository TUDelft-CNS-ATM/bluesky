from PyQt5.QtCore import Qt, QEvent, QTimer, pyqtSlot
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication
from PyQt5.QtOpenGL import QGLWidget, QGLFormat
import OpenGL.GL as gl
import numpy as np
from glob import glob
from math import sin, cos, radians
from ctypes import c_float, c_int, c_void_p, Structure
from glhelpers import BlueSkyProgram, RenderObject, Font, UniformBuffer, load_texture, update_buffer, rect

white                = (255, 255, 255)
black                = (0,   0,   0)
yellow               = (255, 255, 0)
green                = (0,   255, 0)
lightblue2           = (148, 178, 235)
lightblue3           = (220, 250, 255)
VERTEX_IS_LATLON, VERTEX_IS_METERS, VERTEX_IS_SCREEN, VERTEX_IS_GLXY = list(range(4))


def load_lcd_font():
    files = sorted(glob('mcp_font/*.png'))
    img          = QImage(files[0])
    imgsize      = (img.width(), img.height())
    # Set-up the texture array
    tex_id = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D_ARRAY, tex_id)
    gl.glTexImage3D(gl.GL_TEXTURE_2D_ARRAY, 0, gl.GL_RGBA8, imgsize[0], imgsize[1], len(files), 0, gl.GL_BGRA, gl.GL_UNSIGNED_BYTE, None)
    gl.glTexParameteri(gl.GL_TEXTURE_2D_ARRAY, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D_ARRAY, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    gl.glTexParameterf(gl.GL_TEXTURE_2D_ARRAY, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_BORDER)
    gl.glTexParameterf(gl.GL_TEXTURE_2D_ARRAY, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_BORDER)

    for i, fname in enumerate(files):
        img = QImage(fname).convertToFormat(QImage.Format_ARGB32)
        ptr = c_void_p(int(img.constBits()))
        gl.glTexSubImage3D(gl.GL_TEXTURE_2D_ARRAY, 0, 0, 0, i, imgsize[0], imgsize[1], 1, gl.GL_BGRA, gl.GL_UNSIGNED_BYTE, ptr)

    return tex_id


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


def check_knob(px, py):
    # knob name : xmin xmax ymin ymax
    knobs = {
        'COURSEL' : (-0.936, -0.85,  -0.3, 0.18),
        'SPD' :     (-0.52,  -0.454, -0.5, -0.1),
        'HDG' :     (-0.24,  -0.17,  -0.21, 0.2),
        'ALT' :     ( 0.1,    0.166, -0.25, 0.2),
        'VS' :      ( 0.432,  0.476, -0.825, 0.2),
        'COURSER' : ( 0.84,   0.92,  -0.325, 0.13)
    }
    for name, dims in knobs.items():
        if dims[0] <= px <= dims[1] and dims[2] <= py <= dims[3]:
            return name[:6], (0.5 * (dims[0] + dims[1]), 0.5 * (dims[2] + dims[3]))
    return None, None


def check_btn(px, py):
    btns = [
        ('N1', (-0.735, -0.675, -0.84, -0.415)),
        ('SPD', (-0.645, -0.585, -0.84, -0.415)),
        ('LVLCHG', (-0.375, -0.315, -0.84, -0.415)),
        ('HDGSEL', (-0.23, -0.17, -0.84, -0.415)),
        ('APP', (-0.093, -0.033, -0.84, -0.415)),
        ('ALTHLD', (-0.098, 0.158, -0.84, -0.415)),
        ('VS', (0.2, 0.26, -0.84, -0.415)),
        ('VNAV', (-0.373, -0.303, 0.41, 0.85)),
        ('LNAV', (-0.093, -0.033, 0.425, 0.835)),
        ('VORLOC', (-0.093, -0.033, -0.215, 0.175)),
        ('CMDA', (0.575, 0.643, 0.2, 0.625)),
        ('CWSA', (0.575, 0.643, -0.44, -0.025)),
        ('CMDB', (0.683, 0.743, 0.2, 0.625)),
        ('CWSB', (0.683, 0.743, -0.44, -0.025))
    ]
    for i, (name, dims) in enumerate(btns):
        if dims[0] <= px <= dims[1] and dims[2] <= py <= dims[3]:
            return name, i
    return None, None


class BlipDriver(QGLWidget):
    def __init__(self, parent=None):
        super(BlipDriver, self).__init__(parent=parent)
        self.initialized = False
        self.width       = 800
        self.height      = 800
        self.ndviewport  = (100, 200, 600, 600)
        self.mcpviewport = (0, 0, 800, 200)
        self.btn_state   = [False] * 14
        self.resize(800, 800)
        self.setMouseTracking(True)
        self.drag_start  = (0, 0)
        self.btn_pressed = None
        self.selValues   = 5 * [0]  # [0:courseLeftRight, 1:spd, 2:hdg, 3:alt, 4:vs]
        self.rate        = 0.0
        self.remainder   = 0.0
        self.updownpos   = None

    def create_objects(self):
        self.mcp         = RenderObject(gl.GL_TRIANGLES, vertex=np.array(rect(-1, -1, 2, 2), dtype=np.float32))
        self.mcp_texture = load_texture('mcp737.png')
        self.btn_tex     = load_texture('btn_led.png')

        self.lcd_tex     = load_lcd_font()

        v_mcp_text = []
        for pos in [0.0, 0.03, 0.06,
                    0.31, 0.34, 0.37, 0.4, 0.43,
                    0.644, 0.674, 0.704,
                    0.955, 0.985, 1.015, 1.045, 1.075,
                    1.22, 1.25, 1.28, 1.31, 1.34,
                    1.69, 1.72, 1.75]:
            v, t = Font.char(-0.886 + pos, 0.45, 0.03, 0.25)
            v_mcp_text += v

        self.mcp_text       = RenderObject(gl.GL_TRIANGLES, vertex=np.array(v_mcp_text, dtype=np.float32))
        self.lcd_charcoords = np.zeros(24 * 6, dtype=np.float32)
        self.lcdbuf         = self.mcp_text.bind_attrib(1, 1, self.lcd_charcoords, datatype=gl.GL_FLOAT)

        btn_leds = []
        for pos in [(-0.74, -0.75), (-0.645, -0.75), (-0.37, -0.75), (-0.232, -0.75),
                    (-0.09, -0.75), (0.105, -0.75), (0.2, -0.75),
                    (-0.37, 0.5), (-0.09, 0.5), (-0.09, -0.125),
                     (0.575, 0.34), (0.575, -0.34),
                     (0.684, 0.34), (0.684, -0.34)]:
            btn_leds += rect(pos[0], pos[1], 0.055, 0.075)
        btn_color = np.zeros(14 * 6 * 4, dtype=np.uint8)
        self.btn_leds = RenderObject(gl.GL_TRIANGLES, vertex=np.array(btn_leds, dtype=np.float32), color=btn_color)

        # Button up/down indicator
        w, h, offset = 0.04, 0.16, 0.04
        triangles = np.array([-w, offset,  0.0, offset + h, w, offset,
                     -w, -offset, 0.0, -offset - h, w, -offset], dtype=np.float32)
        col_triangles = np.array(6 * [255, 255, 255, 180], dtype=np.uint8)
        self.updown = RenderObject(gl.GL_TRIANGLES, vertex=triangles, color=col_triangles)

        # Use the same font as the radarwidget
        self.font = Font()
        self.font.create_font_array('../../data/graphics/font/')
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

        self.initialized = True
        # Unbind everything
        RenderObject.unbind_all()

    def setAircraftID(self, ac_id):
        self.ac_id = ac_id
        self.setWindowTitle(ac_id)

    def update_aircraft_data(self, ownid, ownlat, ownlon, owntas, ownhdg, n_aircraft):
        self.globaldata.set_owndata(ownid, ownlat, ownlon, ownhdg)
        self.n_aircraft = n_aircraft

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

        # self.mcp_data       = mcpUBO()
        self.globaldata     = ndUBO()
        self.mcp_col_shader = BlueSkyProgram('mcp.vert', 'color.frag')
        self.mcp_tex_shader = BlueSkyProgram('mcp.vert', 'texture.frag')
        self.mcp_txt_shader = BlueSkyProgram('mcp_text.vert', 'mcp_text.frag')

        self.color_shader   = BlueSkyProgram('normal.vert', 'color.frag')
        self.text_shader    = BlueSkyProgram('text.vert', 'text.frag')
        self.text_shader.bind_uniform_buffer('global_data', self.globaldata)

        self.create_objects()
        self.update_lcd()

    def resizeGL(self, width, height):
        pixel_ratio = self.devicePixelRatio()

        self.width, self.height = width / pixel_ratio, height / pixel_ratio
        hmcp = height / 10 * 2
        # paint ND within the largest possible rectangular area in the window
        wnd = hnd = min(width, height - hmcp)
        xnd = max(0, (width - wnd) / 2)
        ynd = hmcp + max(0, (height - hmcp - hnd) / 2)
        self.mcpviewport = (0, 0, width, hmcp)
        self.ndviewport = (xnd, ynd, wnd, hnd)

    def paintGL(self):
        """Paint the scene."""
        # pass if the framebuffer isn't complete yet or if not initialized
        if not (gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) == gl.GL_FRAMEBUFFER_COMPLETE and self.isVisible() and self.initialized):
            return

        gl.glClearColor(0, 0, 0, 0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        # Set the viewport and clear the framebuffer

        # --- Draw the ND in its viewport ----
        gl.glViewport(*self.ndviewport)
        self.color_shader.use()
        self.arcs.draw()

        self.ownship.draw()
        self.mask.draw()
        self.edge.draw()

        self.ticks.draw()

        # Select the text shader
        self.text_shader.use()
        self.font.use()
        self.font.set_block_size((0, 0))
        self.globaldata.set_vertex_modifiers(VERTEX_IS_GLXY, True)
        self.ticklbls.draw()

        self.globaldata.set_vertex_modifiers(VERTEX_IS_GLXY, False)
        self.spdlabel_text.draw()
        self.spdlabel_val.draw()

        # --- Draw the MCP in its viewport ---
        gl.glViewport(*self.mcpviewport)
        self.mcp_tex_shader.use()
        gl.glActiveTexture(gl.GL_TEXTURE0 + 0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.mcp_texture)
        self.mcp.draw()

        gl.glBindTexture(gl.GL_TEXTURE_2D, self.btn_tex)
        self.btn_leds.draw()

        if self.updownpos is not None:
            self.mcp_col_shader.use()
            gl.glVertexAttrib2f(2, *self.updownpos)
            self.updown.draw()
            gl.glVertexAttrib2f(2, 0.0, 0.0)

        self.mcp_txt_shader.use()
        gl.glActiveTexture(gl.GL_TEXTURE0 + 0)
        gl.glBindTexture(gl.GL_TEXTURE_2D_ARRAY, self.lcd_tex)
        self.mcp_text.draw()

        # Unbind everything
        RenderObject.unbind_all()
        gl.glUseProgram(0)

    def update_lcd(self):
        # [0:courseLeftRight, 1:spd, 2:hdg, 3:alt, 4:vs]
        val = self.selValues
        chars = '%03d' % (int(val[0]) % 360)
        chars += '% 5d' % max(0, val[1])
        chars += '%03d' % (int(val[2]) % 360)
        chars += '% 5d' % min(99999, max(0, val[3]))
        chars += '     ' if val[4] == 0 else '% 5d' % min(9999, max(-9999, val[4]))
        chars += '%03d' % (int(val[0]) % 360)

        indices = []
        for c in chars:
            if c == ' ':
                indices += [0] * 6
            elif c == '-':
                indices += [1] * 6
            elif c == '.':
                indices += [2] * 6
            else:
                indices += [int(c) + 3] * 6

        update_buffer(self.lcdbuf, np.array(indices, dtype=np.float32))

    def event(self, event):
        if not self.initialized:
            return super(BlipDriver, self).event(event)

        self.makeCurrent()
        if event.type() in [QEvent.MouseMove, QEvent.MouseButtonPress, QEvent.MouseButtonRelease]:
            px = 2.0 * (event.x()) / self.width - 1.0
            py = 10.0 * (self.height - event.y()) / self.height - 1.0

        if event.type() == QEvent.MouseButtonPress and event.button() & Qt.LeftButton:
            self.btn_pressed, _ = check_knob(px, py)
            if self.btn_pressed is not None:
                self.drag_start = event.x(), event.y()
                QTimer.singleShot(100, self.updateAPValues)

        elif event.type() == QEvent.MouseButtonRelease and event.button() & Qt.LeftButton:
            # print px, py
            self.btn_pressed = None
            col_triangles = 6 * [255, 255, 255, 180]
            update_buffer(self.updown.colorbuf, np.array(col_triangles, dtype=np.uint8))
            # MCP button clicked?
            name, idx = check_btn(px, py)
            if name is not None:
                self.btn_state[idx] = not self.btn_state[idx]

            btn_color = []
            for b in self.btn_state:
                btn_color += 24 * [255] if b else 24 * [0]
            update_buffer(self.btn_leds.colorbuf, np.array(btn_color, dtype=np.uint8))
        elif event.type() == QEvent.MouseMove:
            if event.buttons() & Qt.LeftButton and self.btn_pressed is not None:
                self.rate = float(self.drag_start[1] - event.y()) / self.height
                if self.rate > 1e-2:
                    col_triangles = 3 * [255, 140, 0, 255] + 3 * [255, 255, 255, 180]
                elif self.rate < 1e-2:
                    col_triangles = 3 * [255, 255, 255, 180] + 3 * [255, 140, 0, 255]
                else:
                    col_triangles = 6 * [255, 255, 255, 180]
                update_buffer(self.updown.colorbuf, np.array(col_triangles, dtype=np.uint8))
                return True
            # Mouse-over for knobs
            name, self.updownpos = check_knob(px, py)

            # ismcp = float(self.height - event.y()) / self.height <= 0.2

        return super(BlipDriver, self).event(event)

    @pyqtSlot()
    def updateAPValues(self):
        if self.btn_pressed == 'COURSE':
            val = self.remainder + 15 * self.rate
            self.selValues[0] += int(val)
            self.remainder = val - int(val)

        if self.btn_pressed == 'SPD':
            val = self.remainder + 20 * self.rate
            self.selValues[1] += int(val)
            self.remainder = val - int(val)

        if self.btn_pressed == 'HDG':
            val = self.remainder + 15 * self.rate
            self.selValues[2] += int(val)
            self.remainder = val - int(val)

        if self.btn_pressed == 'ALT':
            if abs(self.rate) > 0.06:
                increment = 1000
                val = self.remainder + min(2.0, abs(self.rate) / 0.04) * np.sign(self.rate)
            elif abs(self.rate) > 0.02:
                increment = 100
                val = self.remainder + self.rate / 0.04
            else:
                increment = 10
                val = self.remainder + self.rate / 0.02

            self.selValues[3] += int(val) * increment
            self.remainder = val - int(val)

        if self.btn_pressed == 'VS':
            if abs(self.rate) > 0.2:
                increment = 1000
                val = self.remainder + min(2.0, abs(self.rate) / 0.04) * np.sign(self.rate)
            elif abs(self.rate) > 0.05:
                increment = 100
                val = self.remainder + self.rate / 0.04
            else:
                increment = 10
                val = self.remainder + self.rate / 0.02

            self.selValues[4] += int(val) * increment
            self.remainder = val - int(val)

        self.update_lcd()
        if self.btn_pressed is not None:
            QTimer.singleShot(200, self.updateAPValues)


if __name__ == "__main__":
    qapp = QApplication([])

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

    blip = BlipDriver()
    blip.show()

    gltimer          = QTimer(qapp)
    gltimer.timeout.connect(blip.updateGL)
    gltimer.start(50)

    qapp.exec_()
