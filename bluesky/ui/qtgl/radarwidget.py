try:
    from PyQt4.QtCore import Qt, qCritical, QTimer
    from PyQt4.QtOpenGL import QGLWidget
    QT_VERSION = 4
except ImportError:
    from PyQt5.QtCore import Qt, qCritical, QTimer
    from PyQt5.QtOpenGL import QGLWidget
    QT_VERSION = 5
from math import sqrt, cos, radians, degrees
import numpy as np
import OpenGL.GL as gl
from ctypes import c_float, c_int, Structure

# Local imports
from ...tools.aero import ft, nm, kts
from glhelpers import BlueSkyProgram, RenderObject, TextObject, update_buffer, UniformBuffer
from uievents import PanZoomEvent, PanZoomEventType
from ...settings import text_size, apt_size, wpt_size, ac_size, font_family, font_weight, text_texture_size


# Static defines
MAX_NAIRCRAFT        = 10000
MAX_NCONFLICTS       = 1000
MAX_ROUTE_LENGTH     = 100
MAX_POLYGON_SEGMENTS = 100

REARTH_INV           = 1.56961231e-7

# Colors
red                  = (1.0, 0.0, 0.0)
green                = (0.0, 1.0, 0.0)
blue                 = (0.0, 0.0, 1.0)
lightblue            = (0.0, 0.8, 1.0)
lightblue2           = (0.33, 0.33, 0.45)
lightblue3           = (0.58, 0.7, 0.92)
lightblue4           = (0.86, 0.98, 1.0)
cyan                 = (0.0, 1.0, 0.0)
amber                = (1.0, 0.6, 0.0)
magenta              = (1.0, 0.0, 1.0)
grey                 = (0.4, 0.4, 0.4)

VERTEX_IS_LATLON, VERTEX_IS_METERS, VERTEX_IS_SCREEN = range(3)
ATTRIB_VERTEX, ATTRIB_TEXCOORDS, ATTRIB_LAT, ATTRIB_LON, ATTRIB_ORIENTATION, ATTRIB_COLOR, ATTRIB_TEXDEPTH = range(7)
ATTRIB_LAT1, ATTRIB_LON1, ATTRIB_ALT1, ATTRIB_TAS1, ATTRIB_TRK1, ATTRIB_LAT0, ATTRIB_LON0, ATTRIB_ALT0, ATTRIB_TAS0, ATTRIB_TRK0 = range(10)


class radarUBO(UniformBuffer):
    class Data(Structure):
        _fields_ = [("wrapdir", c_int), ("wraplon", c_float), ("panlat", c_float), ("panlon", c_float),
        ("zoom", c_float), ("screen_width", c_int), ("screen_height", c_int), ("vertex_scale_type", c_int)]

    data = Data()

    def __init__(self):
        super(radarUBO, self).__init__(self.data)

    def set_wrap(self, wraplon, wrapdir):
        self.data.wrapdir = wrapdir
        self.data.wraplon = wraplon

    def set_pan_and_zoom(self, panlat, panlon, zoom):
        self.data.panlat = panlat
        self.data.panlon = panlon
        self.data.zoom   = zoom

    def set_win_width_height(self, w, h):
        self.data.screen_width  = w
        self.data.screen_height = h

    def enable_wrap(self, flag=True):
        if not flag:
            wrapdir = self.data.wrapdir
            self.data.wrapdir = 0
            self.update(0, 4)
            self.data.wrapdir = wrapdir
        else:
            self.update(0, 4)

    def set_vertex_scale_type(self, vertex_scale_type):
        self.data.vertex_scale_type = vertex_scale_type
        self.update()


class RadarWidget(QGLWidget):
    show_map = show_coast = show_traf = show_pz = show_lbl = show_wpt = show_apt = True
    vcount_circle = 36
    width = height = 600
    viewport = (0, 0, width, height)
    panlat = 51.5
    panlon = 6.5
    zoom = 0.5
    ar = 1.0
    flat_earth = 1.0
    wraplon = int(-999)
    wrapdir = int(0)
    max_texture_size = 0

    do_text = True
    invalid_count = 0

    def __init__(self, navdb, shareWidget=None):
        super(RadarWidget, self).__init__(shareWidget=shareWidget)
        self.setAttribute(Qt.WA_AcceptTouchEvents, True)
        self.grabGesture(Qt.PanGesture)
        self.grabGesture(Qt.PinchGesture)
        self.grabGesture(Qt.SwipeGesture)

        # The number of aircraft in the simulation
        self.map_texture = 0
        self.naircraft   = 0
        self.nwaypoints  = 0
        self.nairports   = 0
        self.route_acidx = -1
        self.ssd_ownship = np.array([], dtype=np.uint16)
        self.ssd_all     = False
        self.navdb       = navdb

        self.initialized = False

    def create_objects(self):
        if not self.isValid():
            self.invalid_count += 1
            print 'Radarwidget: Context not valid in create_objects, count=%d' % self.invalid_count
            QTimer.singleShot(100, self.create_objects)
            return

        # Make the radarwidget context current, necessary when create_objects is not called from initializeGL
        self.makeCurrent()

        # Initialize font for radar view with specified settings
        TextObject.create_font_array(char_height=text_texture_size, font_family=font_family, font_weight=font_weight)
        # Load and bind world texture
        max_texture_size = gl.glGetIntegerv(gl.GL_MAX_TEXTURE_SIZE)
        print 'Maximum supported texture size: %d' % max_texture_size
        for i in [16384, 8192, 4096]:
            if max_texture_size >= i:
                fname = 'data/graphics/world.%dx%d.dds' % (i, i / 2)
                print 'Loading texture ' + fname
                self.map_texture = self.bindTexture(fname)
                break

        # Create initial empty buffers for aircraft position, orientation, label, and color
        self.achdgbuf    = RenderObject.create_empty_buffer(MAX_NAIRCRAFT * 4, usage=gl.GL_STREAM_DRAW)
        self.aclatbuf    = RenderObject.create_empty_buffer(MAX_NAIRCRAFT * 4, usage=gl.GL_STREAM_DRAW)
        self.aclonbuf    = RenderObject.create_empty_buffer(MAX_NAIRCRAFT * 4, usage=gl.GL_STREAM_DRAW)
        self.acaltbuf    = RenderObject.create_empty_buffer(MAX_NAIRCRAFT * 4, usage=gl.GL_STREAM_DRAW)
        self.actasbuf    = RenderObject.create_empty_buffer(MAX_NAIRCRAFT * 4, usage=gl.GL_STREAM_DRAW)
        self.accolorbuf  = RenderObject.create_empty_buffer(MAX_NAIRCRAFT * 12, usage=gl.GL_STREAM_DRAW)
        self.aclblbuf    = RenderObject.create_empty_buffer(MAX_NAIRCRAFT * 15, usage=gl.GL_STREAM_DRAW)
        self.confcpabuf  = RenderObject.create_empty_buffer(MAX_NCONFLICTS * 8, usage=gl.GL_STREAM_DRAW)
        self.polyprevbuf = RenderObject.create_empty_buffer(MAX_POLYGON_SEGMENTS * 8, usage=gl.GL_DYNAMIC_DRAW)
        self.routebuf    = RenderObject.create_empty_buffer(MAX_ROUTE_LENGTH * 8, usage=gl.GL_DYNAMIC_DRAW)

        # ------- Map ------------------------------------
        self.map = RenderObject(gl.GL_TRIANGLE_FAN, vertex_count=4)
        mapvertices = np.array([(-90.0, 540.0), (-90.0, -540.0), (90.0, -540.0), (90.0, 540.0)], dtype=np.float32)
        texcoords = np.array([(1, 3), (1, 0), (0, 0), (0, 3)], dtype=np.float32)
        self.map.bind_attrib(ATTRIB_VERTEX, 2, mapvertices)
        self.map.bind_attrib(ATTRIB_TEXCOORDS, 2, texcoords)

        # ------- Coastlines -----------------------------
        self.coastlines = RenderObject(gl.GL_LINES)
        coastvertices, coastindices = load_coast_data()
        self.coastlines.bind_attrib(ATTRIB_VERTEX, 2, coastvertices)
        self.coastlines.bind_attrib(ATTRIB_COLOR, 3, np.array(lightblue2, dtype=np.float32), instance_divisor=1)
        self.vcount_coast = len(coastvertices)
        self.coastindices = coastindices
        del coastvertices

        # ------- Runways --------------------------------
        self.runways = RenderObject(gl.GL_TRIANGLES)
        rwy_vertices = load_rwy_data()
        self.runways.bind_attrib(ATTRIB_VERTEX, 2, rwy_vertices)
        self.runways.bind_attrib(ATTRIB_COLOR, 3, np.array(grey, dtype=np.float32), instance_divisor=1)
        self.runways.set_vertex_count(len(rwy_vertices)/2)
        del rwy_vertices

        # Polygon preview object
        self.polyprev = RenderObject(gl.GL_LINE_LOOP)
        self.polyprev.bind_attrib(ATTRIB_VERTEX, 2, self.polyprevbuf)
        self.polyprev.bind_attrib(ATTRIB_COLOR, 3, np.array(lightblue, dtype=np.float32), instance_divisor=1)

        # ------- SSD object -----------------------------
        self.ssd = RenderObject(gl.GL_POINTS)
        self.ssd.bind_attrib(ATTRIB_LAT0, 1, self.aclatbuf)
        self.ssd.bind_attrib(ATTRIB_LON0, 1, self.aclonbuf)
        self.ssd.bind_attrib(ATTRIB_ALT0, 1, self.acaltbuf)
        self.ssd.bind_attrib(ATTRIB_TAS0, 1, self.actasbuf)
        self.ssd.bind_attrib(ATTRIB_TRK0, 1, self.achdgbuf)
        self.ssd.bind_attrib(ATTRIB_LAT1, 1, self.aclatbuf, instance_divisor=1)
        self.ssd.bind_attrib(ATTRIB_LON1, 1, self.aclonbuf, instance_divisor=1)
        self.ssd.bind_attrib(ATTRIB_ALT1, 1, self.acaltbuf, instance_divisor=1)
        self.ssd.bind_attrib(ATTRIB_TAS1, 1, self.actasbuf, instance_divisor=1)
        self.ssd.bind_attrib(ATTRIB_TRK1, 1, self.achdgbuf, instance_divisor=1)

        # ------- Circle ---------------------------------
        # Create a new VAO (Vertex Array Object) and bind it
        self.protectedzone = RenderObject(gl.GL_LINE_LOOP, vertex_count=self.vcount_circle)
        circlevertices = np.transpose(np.array((5.0*nm*np.cos(np.linspace(0.0, 2.0*np.pi, self.vcount_circle)), 5.0*nm*np.sin(np.linspace(0.0, 2.0*np.pi, self.vcount_circle))), dtype=np.float32))
        self.protectedzone.bind_attrib(ATTRIB_VERTEX, 2, circlevertices)
        self.protectedzone.bind_attrib(ATTRIB_LAT, 1, self.aclatbuf, instance_divisor=1)
        self.protectedzone.bind_attrib(ATTRIB_LON, 1, self.aclonbuf, instance_divisor=1)
        self.protectedzone.bind_attrib(ATTRIB_COLOR, 3, self.accolorbuf, instance_divisor=1)

        # ------- A/C symbol -----------------------------
        self.ac_symbol = RenderObject(gl.GL_TRIANGLE_FAN, vertex_count=4)
        acvertices = np.array([(0.0, 0.5 * ac_size), (-0.5 * ac_size, -0.5 * ac_size), (0.0, -0.25 * ac_size), (0.5 * ac_size, -0.5 * ac_size)], dtype=np.float32)
        self.ac_symbol.bind_attrib(ATTRIB_VERTEX, 2, acvertices)
        self.ac_symbol.bind_attrib(ATTRIB_LAT, 1, self.aclatbuf, instance_divisor=1)
        self.ac_symbol.bind_attrib(ATTRIB_LON, 1, self.aclonbuf, instance_divisor=1)
        self.ac_symbol.bind_attrib(ATTRIB_ORIENTATION, 1, self.achdgbuf, instance_divisor=1)
        self.ac_symbol.bind_attrib(ATTRIB_COLOR, 3, self.accolorbuf, instance_divisor=1)
        self.aclabels = TextObject()
        self.aclabels.prepare_text_instanced(self.aclblbuf, self.aclatbuf, self.aclonbuf, (6, 3), self.accolorbuf, text_size=text_size, vertex_offset=(ac_size, -0.5 * ac_size))

        # ------- Conflict CPA lines ---------------------
        self.cpalines = RenderObject(gl.GL_LINES)
        self.cpalines.bind_attrib(ATTRIB_VERTEX, 2, self.confcpabuf)
        self.cpalines.bind_attrib(ATTRIB_COLOR, 3, np.array(amber, dtype=np.float32), instance_divisor=1)

        # ------- Aircraft Route -------------------------
        self.route = RenderObject(gl.GL_LINE_STRIP)
        self.route.bind_attrib(ATTRIB_VERTEX, 2, self.routebuf)
        self.route.bind_attrib(ATTRIB_COLOR, 3, np.array(magenta, dtype=np.float32), instance_divisor=1)

        # ------- Waypoints ------------------------------
        self.nwaypoints = len(self.navdb.wplat)
        self.waypoints = RenderObject(gl.GL_LINE_LOOP, vertex_count=3, n_instances=self.nwaypoints)
        wptvertices = np.array([(0.0, 0.5 * wpt_size), (-0.5 * wpt_size, -0.5 * wpt_size), (0.5 * wpt_size, -0.5 * wpt_size)], dtype=np.float32)  # a triangle
        self.waypoints.bind_attrib(ATTRIB_VERTEX, 2, wptvertices)
        self.wptlatbuf = self.waypoints.bind_attrib(ATTRIB_LAT, 1, np.array(self.navdb.wplat, dtype=np.float32), instance_divisor=1)
        self.wptlonbuf = self.waypoints.bind_attrib(ATTRIB_LON, 1, np.array(self.navdb.wplon, dtype=np.float32), instance_divisor=1)
        self.wptlabels = TextObject()
        wptids = ''
        for wptid in self.navdb.wpid:
            wptids += wptid.ljust(5)
        self.wptlabels.prepare_text_instanced(np.array(wptids, dtype=np.string_), self.wptlatbuf, self.wptlonbuf, (5, 1), text_size=text_size, vertex_offset=(wpt_size, 0.5 * wpt_size))
        del wptids

        # ------- Airports -------------------------------
        self.nairports = len(self.navdb.aplat)
        self.airports = RenderObject(gl.GL_LINE_LOOP, vertex_count=4, n_instances=self.nairports)
        aptvertices = np.array([(-0.5 * apt_size, -0.5 * apt_size), (0.5 * apt_size, -0.5 * apt_size), (0.5 * apt_size, 0.5 * apt_size), (-0.5 * apt_size, 0.5 * apt_size)], dtype=np.float32)  # a square
        self.airports.bind_attrib(ATTRIB_VERTEX, 2, aptvertices)
        indices = self.navdb.aptype.argsort()
        aplat   = np.array(self.navdb.aplat[indices], dtype=np.float32)
        aplon   = np.array(self.navdb.aplon[indices], dtype=np.float32)
        aptypes = self.navdb.aptype[indices]
        apnames = np.array(self.navdb.apid)
        apnames = apnames[indices]
        # The number of large, large+med, and large+med+small airports
        self.nairports = [aptypes.searchsorted(2), aptypes.searchsorted(3), self.nairports]

        self.aptlatbuf = self.airports.bind_attrib(ATTRIB_LAT, 1, aplat, instance_divisor=1)
        self.aptlonbuf = self.airports.bind_attrib(ATTRIB_LON, 1, aplon, instance_divisor=1)
        self.aptlabels = TextObject()
        aptids = ''
        for aptid in apnames:
            aptids += aptid.ljust(4)
        self.aptlabels.prepare_text_instanced(np.array(aptids, dtype=np.string_), self.aptlatbuf, self.aptlonbuf, (4, 1), text_size=text_size, vertex_offset=(apt_size, 0.5 * apt_size))
        del aptids

        # Create a dictionary that can hold a named list of shapes that can be added through the stack
        self.polys = dict()

        # Unbind VAO, VBO
        RenderObject.unbind_all()

        # Set initial values for the global uniforms
        self.globaldata.set_wrap(self.wraplon, self.wrapdir)
        self.globaldata.set_pan_and_zoom(self.panlat, self.panlon, self.zoom)

        self.initialized = True

    def initializeGL(self):
        """Initialize OpenGL, VBOs, upload data on the GPU, etc."""

        # First check for supported GL version
        gl_version = float(gl.glGetString(gl.GL_VERSION)[:3])
        if gl_version < 3.3:
            print('OpenGL context created with GL version %.1f' % gl_version)
            qCritical('Your system reports that it supports OpenGL up to version %.1f. The minimum requirement for BlueSky is OpenGL 3.3. \
                Generally, AMD/ATI/nVidia cards from 2008 and newer support OpenGL 3.3, and Intel integrated graphics from the Haswell \
                generation and newer. If you think your graphics system should be able to support GL>=3.3 please open an issue report \
                on the BlueSky Github page (https://github.com/ProfHoekstra/bluesky/issues)' % gl_version)
            return

        # background color
        gl.glClearColor(0, 0, 0, 0)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        self.globaldata = radarUBO()

        try:
            # Compile shaders and link color shader program
            self.color_shader = BlueSkyProgram('data/graphics/shaders/radarwidget-normal.vert', 'data/graphics/shaders/radarwidget-color.frag')
            self.color_shader.bind_uniform_buffer('global_data', self.globaldata)

            # Compile shaders and link texture shader program
            self.texture_shader = BlueSkyProgram('data/graphics/shaders/radarwidget-normal.vert', 'data/graphics/shaders/radarwidget-texture.frag')
            self.texture_shader.bind_uniform_buffer('global_data', self.globaldata)

            # Compile shaders and link text shader program
            self.text_shader = BlueSkyProgram('data/graphics/shaders/radarwidget-text.vert', 'data/graphics/shaders/radarwidget-text.frag')
            self.text_shader.bind_uniform_buffer('global_data', self.globaldata)
            TextObject.init_shader(self.text_shader)

            self.ssd_shader = BlueSkyProgram('data/graphics/shaders/ssd.vert', 'data/graphics/shaders/ssd.frag', 'data/graphics/shaders/ssd.geom')
            self.ssd_shader.bind_uniform_buffer('global_data', self.globaldata)
            self.ssd_shader.loc_vlimits = gl.glGetUniformLocation(self.ssd_shader.program, 'Vlimits')

        except RuntimeError as e:
            qCritical('Error compiling shaders in radarwidget: ' + e.args[0])
            return

        # create all vertex array objects
        self.create_objects()

    def paintGL(self):
        """Paint the scene."""
        # pass if the framebuffer isn't complete yet or if not initialized
        if not (gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) == gl.GL_FRAMEBUFFER_COMPLETE and self.initialized and self.isVisible()):
            return

        # Set the viewport and clear the framebuffer
        gl.glViewport(*self.viewport)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        # Send the (possibly) updated global uniforms to the buffer
        self.globaldata.set_vertex_scale_type(VERTEX_IS_LATLON)

        # --- DRAW THE MAP AND COASTLINES ---------------------------------------------
        # Map and coastlines: don't wrap around in the shader
        self.globaldata.enable_wrap(False)

        if self.show_map:
            # Select the texture shader
            self.texture_shader.use()

            # Draw map texture
            gl.glActiveTexture(gl.GL_TEXTURE0 + 0)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.map_texture)
            self.map.draw()

        # Select the non-textured shader
        self.color_shader.use()

        # Draw coastlines
        if self.show_coast:
            if self.wrapdir == 0:
                # Normal case, no wrap around
                self.coastlines.draw(first_vertex=0, vertex_count=self.vcount_coast)
            else:
                self.coastlines.bind()
                wrapindex = np.uint32(self.coastindices[int(self.wraplon)+180])
                if self.wrapdir == 1:
                    gl.glVertexAttrib1f(ATTRIB_LON, 360.0)
                    self.coastlines.draw(first_vertex=0, vertex_count=wrapindex)
                    gl.glVertexAttrib1f(ATTRIB_LON, 0.0)
                    self.coastlines.draw(first_vertex=wrapindex, vertex_count=self.vcount_coast - wrapindex)
                else:
                    gl.glVertexAttrib1f(ATTRIB_LON, -360.0)
                    self.coastlines.draw(first_vertex=wrapindex, vertex_count=self.vcount_coast - wrapindex)
                    gl.glVertexAttrib1f(ATTRIB_LON, 0.0)
                    self.coastlines.draw(first_vertex=0, vertex_count=wrapindex)

        # --- DRAW PREVIEW SHAPE (WHEN AVAILABLE) -----------------------------
        self.polyprev.draw()

        # --- DRAW CUSTOM SHAPES (WHEN AVAILABLE) -----------------------------
        for i in self.polys.iteritems():
            i[1].draw()

        # --- DRAW THE SELECTED AIRCRAFT ROUTE (WHEN AVAILABLE) ---------------
        if self.show_traf:
            self.route.draw()

        if self.show_traf:
            self.cpalines.draw()

        if self.zoom >= 1.0:
            self.runways.draw()

        # --- DRAW THE INSTANCED AIRCRAFT SHAPES ------------------------------
        # update wrap longitude and direction for the instanced objects
        self.globaldata.enable_wrap(True)

        # PZ circles only when they are bigger than the A/C symbols
        if self.naircraft > 0 and self.show_traf and self.show_pz and self.zoom >= 0.15:
            self.globaldata.set_vertex_scale_type(VERTEX_IS_METERS)
            self.protectedzone.draw(n_instances=self.naircraft)

        self.globaldata.set_vertex_scale_type(VERTEX_IS_SCREEN)

        # Draw traffic symbols
        if self.naircraft > 0 and self.show_traf:
            self.ac_symbol.draw(n_instances=self.naircraft)

        if self.zoom >= 0.5:
            nairports = self.nairports[2]
            show_wpt = self.show_wpt
        elif self.zoom  >= 0.25:
            nairports = self.nairports[1]
            show_wpt = False
        else:
            nairports = self.nairports[0]
            show_wpt = False

        # Draw waypoint symbols
        if show_wpt:
            self.waypoints.bind()
            gl.glVertexAttrib3f(ATTRIB_COLOR, *lightblue3)
            self.waypoints.draw(n_instances=self.nwaypoints)

        # Draw airport symbols
        if self.show_apt:
            self.airports.bind()
            gl.glVertexAttrib3f(ATTRIB_COLOR, *lightblue3)
            self.airports.draw(n_instances=nairports)

        if self.do_text:
            self.text_shader.use()

            if self.show_apt:
                self.aptlabels.bind()
                gl.glVertexAttrib3f(ATTRIB_COLOR, *lightblue4)
                self.aptlabels.draw(n_instances=nairports)
            if self.zoom >= 1.0 and show_wpt:
                self.wptlabels.bind()
                gl.glVertexAttrib3f(ATTRIB_COLOR, *lightblue4)
                self.wptlabels.draw(n_instances=self.nwaypoints)
            if self.naircraft > 0 and self.show_traf and self.show_lbl:
                self.aclabels.draw(n_instances=self.naircraft)

        # SSD
        if self.ssd_all or len(self.ssd_ownship) > 0:
            self.ssd_shader.use()
            gl.glUniform3f(self.ssd_shader.loc_vlimits, 1e4, 4e4, 200.0)
            if self.ssd_all:
                self.ssd.draw(first_vertex=0, vertex_count=self.naircraft, n_instances=self.naircraft)
            else:
                self.ssd.draw(first_vertex=self.ssd_ownship[-1], vertex_count=1, n_instances=self.naircraft)

        # Unbind everything
        RenderObject.unbind_all()
        gl.glUseProgram(0)

    def resizeGL(self, width, height):
        """Called upon window resizing: reinitialize the viewport."""
        # update the window size
        # Qt5 supports getting the device pixel ratio, which can be > 1 for HiDPI displays such as Mac Retina screens
        pixel_ratio = 1
        if QT_VERSION >= 5:
            pixel_ratio = self.devicePixelRatio()

        # Calculate zoom so that the window resize doesn't affect the scale, but only enlarges or shrinks the view
        zoom   = float(self.width) / float(width) * pixel_ratio
        origin = (width / 2, height / 2)

        # Update width, height, and aspect ratio
        self.width, self.height = width / pixel_ratio, height / pixel_ratio
        self.ar = float(width) / max(1, float(height))
        self.globaldata.set_win_width_height(self.width, self.height)

        self.viewport = (0, 0, width, height)

        # Update zoom
        self.event(PanZoomEvent(zoom=zoom, origin=origin))

    def update_route_data(self, data):
        self.route_acidx      = data.acidx
        if self.route_acidx >= 0:
            nsegments = len(data.lat)
            self.route.set_vertex_count(nsegments)
            routedata       = np.empty(2 * nsegments, dtype=np.float32)
            routedata[::2]  = data.lat
            routedata[1::2] = data.lon
            update_buffer(self.routebuf, routedata)
        else:
            self.route.set_vertex_count(0)

    def update_aircraft_data(self, data):
        self.naircraft = len(data.lat)
        if self.naircraft > 0:
            # Update data in GPU buffers
            update_buffer(self.aclatbuf, data.lat)
            update_buffer(self.aclonbuf, data.lon)
            update_buffer(self.achdgbuf, data.trk)
            update_buffer(self.acaltbuf, data.alt)
            update_buffer(self.actasbuf, data.tas)

            # CPA lines to indicate conflicts
            ncpalines = len(data.confcpalat)
            cpalines  = np.zeros(4 * ncpalines, dtype=np.float32)
            self.cpalines.set_vertex_count(2 * ncpalines)

            # Labels and colors
            rawlabel = ''
            color    = np.zeros((self.naircraft, 3), dtype=np.float32)
            for i in range(self.naircraft):
                if np.isnan(data.tas[i]):
                    print 'TAS NaN in %d: %s' % (i, data.id[i])
                    data.tas[i] = 0.0

                if np.isnan(data.alt[i]):
                    print 'ALT NaN in %d: %s' % (i, data.id[i])
                    data.alt[i] = 0.0

                rawlabel += '%-6sFL%03d %-6d' % (data.id[i], int(data.alt[i] / ft / 100), int(data.tas[i] / kts))
                confidx = data.iconf[i]
                if confidx >= 0:
                    color[i, :] = amber
                    cpalines[4 * confidx : 4 * confidx + 4] = [ data.lat[i], data.lon[i],
                                                                data.confcpalat[confidx], data.confcpalon[confidx]]
                else:
                    color[i, :] = green

            update_buffer(self.confcpabuf, cpalines)
            update_buffer(self.accolorbuf, color)
            update_buffer(self.aclblbuf, np.array(rawlabel, dtype=np.string_))

            # If there is a visible route, update the start position
            if self.route_acidx >= 0:
                update_buffer(self.routebuf, np.array([data.lat[self.route_acidx], data.lon[self.route_acidx]], dtype=np.float32))

    def show_ssd(self, arg):
        if arg == 'ALL':
            self.ssd_all = True
        elif arg == 'OFF':
            self.ssd_all = False
            self.ssd_ownship = np.array([], dtype=np.uint16)
        else:
            if arg in self.ssd_ownship:
                self.ssd_ownship = np.delete(self.ssd_ownship, np.argmax(self.ssd_ownship == arg))
            else:
                self.ssd_ownship = np.append(self.ssd_ownship, arg)

    def updatePolygon(self, name, data_in):
        self.makeCurrent()
        if name in self.polys:
            if data_in is None:
                del self.polys[name]
            else:
                update_buffer(self.polys[name].vertexbuf, data_in)
                self.polys[name].set_vertex_count(len(data_in) / 2)
        else:
            if data_in is None:
                print "Delete '" + name + "': object not found!"
            else:
                newpoly = RenderObject(gl.GL_LINE_LOOP, vertex_count=len(data_in)/2)
                newpoly.bind_attrib(ATTRIB_VERTEX, 2, data_in)
                newpoly.bind_attrib(ATTRIB_COLOR, 3, np.array(blue, dtype=np.float32), instance_divisor=1)
                self.polys[name] = newpoly

    def previewpoly(self, shape_type, data_in=None):
        if shape_type is None:
            self.polyprev.set_vertex_count(0)
            return
        if shape_type in ['BOX', 'AREA']:
            # For a box (an area is a box) we need to add two additional corners
            data = np.zeros(8, dtype=np.float32)
            data[0:2] = data_in[0:2]
            data[2:4] = data_in[2], data_in[1]
            data[4:6] = data_in[2:4]
            data[6:8] = data_in[0], data_in[3]
        else:
            data = data_in
        update_buffer(self.polyprevbuf, data)
        self.polyprev.set_vertex_count(len(data)/2)

    def pixelCoordsToGLxy(self, x, y):
        """Convert screen pixel coordinates to GL projection coordinates (x, y range -1 -- 1)
        """
        # GL coordinates (x, y range -1 -- 1)
        glx = (float(2.0 * x) / self.width  - 1.0)
        gly = -(float(2.0 * y) / self.height - 1.0)
        return glx, gly

    def pixelCoordsToLatLon(self, x, y):
        """Convert screen pixel coordinates to lat/lon coordinates
        """
        glx, gly = self.pixelCoordsToGLxy(x, y)

        # glxy   = zoom * (latlon - pan)
        # latlon = pan + glxy / zoom
        lat = self.panlat + gly / (self.zoom * self.ar)
        lon = self.panlon + glx / (self.zoom * self.flat_earth)
        return lat, lon

    def event(self, event):
        if event.type() == PanZoomEventType:
            if event.pan is not None:
                # Absolute pan operation
                if event.absolute:
                    self.panlat = event.pan[0]
                    self.panlon = event.pan[1]
                # Relative pan operation
                else:
                    self.panlat += event.pan[0]
                    self.panlon += event.pan[1]

                # Don't pan further than the poles in y-direction
                self.panlat = min(max(self.panlat, -90.0 + 1.0 /
                      (self.zoom * self.ar)), 90.0 - 1.0 / (self.zoom * self.ar))

                # Update flat-earth factor and possibly zoom in case of very wide windows (> 2:1)
                self.flat_earth = np.cos(np.deg2rad(self.panlat))
                self.zoom = max(self.zoom, 1.0 / (180.0 * self.flat_earth))

            if event.zoom is not None:
                if event.absolute:
                    # Limit zoom extents in x-direction to [-180:180], and in y-direction to [-90:90]
                    self.zoom = max(event.zoom, 1.0 / min(90.0 * self.ar, 180.0 * self.flat_earth))
                else:
                    prevzoom = self.zoom
                    glx, gly = self.pixelCoordsToGLxy(event.origin[0], event.origin[1])
                    self.zoom *= event.zoom

                    # Limit zoom extents in x-direction to [-180:180], and in y-direction to [-90:90]
                    self.zoom = max(self.zoom, 1.0 / min(90.0 * self.ar, 180.0 * self.flat_earth))

                    # Correct pan so that zoom actions are around the mouse position, not around 0, 0
                    # glxy / zoom1 - pan1 = glxy / zoom2 - pan2
                    # pan2 = pan1 + glxy (1/zoom2 - 1/zoom1)
                    self.panlon = self.panlon - glx * (1.0 / self.zoom - 1.0 / prevzoom) / self.flat_earth
                    self.panlat = self.panlat - gly * (1.0 / self.zoom - 1.0 / prevzoom) / self.ar

                # Don't pan further than the poles in y-direction
                self.panlat = min(max(self.panlat, -90.0 + 1.0 / (self.zoom * self.ar)), 90.0 - 1.0 / (self.zoom * self.ar))

                # Update flat-earth factor
                self.flat_earth = np.cos(np.deg2rad(self.panlat))
            event.accept()

            # Check for necessity wrap-around in x-direction
            self.wraplon  = -999.9
            self.wrapdir  = 0
            if self.panlon + 1.0 / (self.zoom * self.flat_earth) < -180.0:
                # The left edge of the map has passed the right edge of the screen: we can just change the pan position
                self.panlon += 360.0
            elif self.panlon - 1.0 / (self.zoom * self.flat_earth) < -180.0:
                # The left edge of the map has passed the left edge of the screen: we need to wrap around to the left
                self.wraplon = float(np.ceil(360.0 + self.panlon - 1.0 / (self.zoom * self.flat_earth)))
                self.wrapdir = -1
            elif self.panlon - 1.0 / (self.zoom * self.flat_earth) > 180.0:
                # The right edge of the map has passed the left edge of the screen: we can just change the pan position
                self.panlon -= 360.0
            elif self.panlon + 1.0 / (self.zoom * self.flat_earth) > 180.0:
                # The right edge of the map has passed the right edge of the screen: we need to wrap around to the right
                self.wraplon = float(np.floor(-360.0 + self.panlon + 1.0 / (self.zoom * self.flat_earth)))
                self.wrapdir = 1

            self.globaldata.set_wrap(self.wraplon, self.wrapdir)

            # update pan and zoom on GPU for all shaders
            self.globaldata.set_pan_and_zoom(self.panlat, self.panlon, self.zoom)
            return True

        else:
            return super(RadarWidget, self).event(event)


def load_rwy_data():
    # total number of runways
    nrwy_tot = 33721
    vertices = np.zeros(nrwy_tot * 12, dtype=np.float32)

    with open("data/global/apt.dat", 'r') as f:
        n_rwys = 0
        for line in f:
            line = line.strip()
            if line[:4] == '100 ':
                # This line holds a runway
                n_rwys += 1
                l = line.split()
                width = float(l[1])
                # Only asphalt and concrete runways
                if int(l[2]) > 2:
                    continue
                # rwy_lbl = (l[8], l[17])
                lat0 = float(l[9])
                lon0 = float(l[10])
                lat1 = float(l[18])
                lon1 = float(l[19])
                flat_earth = cos(0.5 * radians(lat0 + lat1))
                lx = lat1 - lat0
                ly = (lon1 - lon0) * flat_earth
                l  = sqrt(lx * lx + ly * ly)
                wx =  ly / l * 0.5 * width
                wy = -lx / l * 0.5 * width
                dlat = degrees(wx * REARTH_INV)
                dlon = degrees(wy * REARTH_INV / flat_earth)
                #d_inv = 1.0 / sqrt(dlat * dlat + dlon * dlon)
                #dlat = degrees(dlat * 0.5 * width * REARTH_INV * d_inv)
                #dlon = degrees(dlon * 0.5 * width * REARTH_INV * d_inv)
                vertices[(n_rwys - 1) * 12:n_rwys * 12] = [
                            lat0 + dlat, lon0 + dlon,
                            lat0 - dlat, lon0 - dlon,
                            lat1 + dlat, lon1 + dlon,
                            lat0 - dlat, lon0 - dlon,
                            lat1 + dlat, lon1 + dlon,
                            lat1 - dlat, lon1 - dlon]
        print '%d runways read' % n_rwys

    return vertices


def load_coast_data():
    """Load static data: coaslines, waypoint and airport database.
    """
    # -------------------------COASTLINE DATA----------------------------------
    # Init geo (coastline)  data and convert pen up/pen down format of
    # coastlines to numpy arrays with lat/lon
    coast = []
    clat = clon = 0.0
    with open("data/global/coastlines.dat", 'r') as f:
        for line in f:
            line = line.strip()
            if not (line == "" or line[0] == '#'):
                arg = line.split()
                if len(arg) == 3:
                    lat, lon = float(arg[1]), float(arg[2])
                    if arg[0] == 'D':
                        coast.append([clat, clon, lat, lon])
                    clat, clon = lat, lon
    # Sort the line segments by longitude of the first vertex
    coastvertices = np.array(
        sorted(coast, key=lambda a_entry: a_entry[1]), dtype=np.float32)
    coastindices = np.zeros(361)
    coastlon = coastvertices[:, 1]
    for i in range(0, 360):
        coastindices[i] = np.searchsorted(coastlon, i - 180) * 2
    coastvertices.resize((coastvertices.size/2, 2))
    del coast
    return coastvertices, coastindices
