try:
    from PyQt4.QtCore import Qt
    from PyQt4.QtOpenGL import QGLWidget, QGLFormat, QGLContext
    QT_VERSION = 4
except ImportError:
    from PyQt5.QtCore import Qt
    from PyQt5.QtOpenGL import QGLWidget, QGLFormat, QGLContext
    QT_VERSION = 5
import numpy as np
import OpenGL.GL as gl
import time

# Local imports
from ...tools.aero import ft, nm, kts
from glhelpers import BlueSkyProgram, RenderObject, TextObject, update_array_buffer
from uievents import PanZoomEvent

VERTEX_IS_LATLON, VERTEX_IS_METERS, VERTEX_IS_SCREEN = range(3)

# Static defines
MAX_NAIRCRAFT = 10000


class RadarWidget(QGLWidget):
    show_traf = show_lbl = show_wpt = show_apt = True
    vcount_circle = 36
    width = height = 600
    pan = [0.0, 0.0]
    zoom = 1.0
    ar = 1.0
    flat_earth = 1.0
    wraplon = int(-999)
    wrapdir = int(0)
    color_wpt = color_apt = (149.0/255.0, 179.0/255.0, 235/255.0)
    color_wptlbl = color_aptlbl = (219.0/255.0, 249.0/255.0, 255/255.0)

    def __init__(self, parent=None):
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

        self.setAttribute(Qt.WA_AcceptTouchEvents, True)
        self.grabGesture(Qt.PanGesture)
        self.grabGesture(Qt.PinchGesture)
        self.grabGesture(Qt.SwipeGesture)
        self.setAutoBufferSwap(False)

        # The number of aircraft in the simulation
        self.naircraft = 0
        self.nwaypoints = 0
        self.nairports = 0

    def create_objects(self):
        # Load and bind world texture
        # self.map_texture = load_texture('data/world.jpg')
        self.map_texture = self.bindTexture('data/graphics/world.16384x8192-mipmap.dds')

        # Create initial empty buffers for aircraft position, orientation, label, and color
        self.achdgbuf = RenderObject.create_empty_buffer(MAX_NAIRCRAFT * 4, usage=gl.GL_STREAM_DRAW)
        self.aclatbuf = RenderObject.create_empty_buffer(MAX_NAIRCRAFT * 4, usage=gl.GL_STREAM_DRAW)
        self.aclonbuf = RenderObject.create_empty_buffer(MAX_NAIRCRAFT * 4, usage=gl.GL_STREAM_DRAW)
        self.accolorbuf = RenderObject.create_empty_buffer(MAX_NAIRCRAFT * 12, usage=gl.GL_STREAM_DRAW)
        self.aclblbuf = RenderObject.create_empty_buffer(MAX_NAIRCRAFT * 15, usage=gl.GL_STREAM_DRAW)

        # ------- Map ------------------------------------
        self.map = RenderObject()
        mapvertices = np.array([(540.0, -90.0), (-540.0, -90.0), (-540.0, 90.0), (540.0, 90.0)], dtype=np.float32)
        texcoords = np.array([(3, 1), (0, 1), (0, 0), (3, 0)], dtype=np.float32)
        self.map.bind_vertex_attribute(mapvertices)
        self.map.bind_texcoords_attribute(texcoords)

        # ------- Circle ---------------------------------
        # Create a new VAO (Vertex Array Object) and bind it
        self.protectedzone = RenderObject()
        circlevertices = np.transpose(np.array((5.0*nm*np.cos(np.linspace(0.0, 2.0*np.pi, self.vcount_circle)), 5.0*nm*np.sin(np.linspace(0.0, 2.0*np.pi, self.vcount_circle))), dtype=np.float32))
        self.protectedzone.bind_vertex_attribute(circlevertices)
        self.protectedzone.bind_lat_attribute(self.aclatbuf)
        self.protectedzone.bind_lon_attribute(self.aclonbuf)
        self.protectedzone.bind_color_attribute(self.accolorbuf)

        # ------- A/C symbol -----------------------------
        self.ac_symbol = RenderObject()
        acvertices = np.array([(0.0, 0.01), (-0.01, -0.01), (0.0, -0.005), (0.01, -0.01)], dtype=np.float32)
        self.ac_symbol.bind_vertex_attribute(acvertices)
        self.ac_symbol.bind_lat_attribute(self.aclatbuf)
        self.ac_symbol.bind_lon_attribute(self.aclonbuf)
        self.ac_symbol.bind_orientation_attribute(self.achdgbuf)
        self.ac_symbol.bind_color_attribute(self.accolorbuf)
        self.aclabels = TextObject()
        self.aclabels.prepare_text_instanced(self.aclblbuf, self.aclatbuf, self.aclonbuf, (6, 3), self.accolorbuf, vertex_offset=(0.02, -0.01))

        # ------- Coastlines -----------------------------
        self.coastlines = RenderObject()
        self.coastlines.bind_vertex_attribute(self.coastvertices)
        coastlinecolor = np.array((84.0/255.0, 84.0/255.0, 114.0/255.0), dtype=np.float32)
        self.coastlines.bind_color_attribute(coastlinecolor)

        # ------- Waypoints ------------------------------
        self.waypoints = RenderObject()
        wptvertices = np.array([(0.0, 0.006), (-0.006, -0.006), (0.006, -0.006)], dtype=np.float32)  # a triangle
        self.waypoints.bind_vertex_attribute(wptvertices)
        self.wptlatbuf = self.waypoints.bind_lat_attribute(self.wptlat)
        self.wptlonbuf = self.waypoints.bind_lon_attribute(self.wptlon)
        self.wptlabels = TextObject()
        self.wptlabels.prepare_text_instanced(self.wptlabeldata, self.wptlatbuf, self.wptlonbuf, (5, 1), vertex_offset=(0.02, 0.01))

        # ------- Airports -------------------------------
        self.airports = RenderObject()
        aptvertices = np.array([(-0.006, -0.006), (0.006, -0.006), (0.006, 0.006), (-0.006, 0.006)], dtype=np.float32)  # a square
        self.airports.bind_vertex_attribute(aptvertices)
        self.aptlatbuf = self.airports.bind_lat_attribute(self.aptlat)
        self.aptlonbuf = self.airports.bind_lon_attribute(self.aptlon)
        self.aptlabels = TextObject()
        self.aptlabels.prepare_text_instanced(self.aptlabeldata, self.aptlatbuf, self.aptlonbuf, (4, 1), vertex_offset=(0.02, 0.01))

        # Unbind VAO, VBO
        RenderObject.unbind_all()

        # Set initial values for the global uniforms
        BlueSkyProgram.set_wrap(self.wraplon, self.wrapdir)
        BlueSkyProgram.set_pan_and_zoom(self.pan[0], self.pan[1], self.zoom)

    def initializeGL(self):
        """Initialize OpenGL, VBOs, upload data on the GPU, etc."""
        # f = open('opengl_test.txt', 'w')
        # f.write('Supported OpenGL version: ' + gl.glGetString(gl.GL_VERSION) + '\n')
        # f.write('Supported GLSL version: ' + gl.glGetString(gl.GL_SHADING_LANGUAGE_VERSION) + '\n')
        # numext = gl.glGetIntegerv(gl.GL_NUM_EXTENSIONS)
        # f.write('Supported OpenGL extensions:' + '\n')
        # extensions = ''
        # for i in range(numext):
        #     extensions += ', ' + gl.glGetStringi(gl.GL_EXTENSIONS, i)
        # f.write(extensions)
        # f.close()
        # return
        # background color
        gl.glClearColor(0, 0, 0, 0)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        # Compile shaders and link color shader program
        self.color = BlueSkyProgram('data/graphics/shaders/shader.vert', 'data/graphics/shaders/color_shader.frag')

        # Compile shaders and link texture shader program
        self.texture = BlueSkyProgram('data/graphics/shaders/shader.vert', 'data/graphics/shaders/texture_shader.frag')

        # Compile shaders and link text shader program
        self.text = BlueSkyProgram('data/graphics/shaders/shader_text.vert', 'data/graphics/shaders/shader_text.frag')
        TextObject.init_shader(self.text)

        # create a vertex array objects
        self.load_data()
        self.create_objects()

    def paintGL(self):
        """Paint the scene."""
        # pass if the framebuffer isn't complete yet
        if not gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) == gl.GL_FRAMEBUFFER_COMPLETE:
            return

        # clear the buffer
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        # Send the (possibly) updated global uniforms to the buffer
        BlueSkyProgram.update_global_uniforms()
        BlueSkyProgram.set_vertex_scale_type(VERTEX_IS_LATLON)
        # Select the texture shader
        self.texture.use()

        # --- DRAW THE MAP AND COASTLINES ---------------------------------------------
        # Map and coastlines: don't wrap around in the shader
        BlueSkyProgram.enable_wrap(False)

        # Draw map texture
        gl.glActiveTexture(gl.GL_TEXTURE0 + 0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.map_texture)
        self.map.draw(gl.GL_TRIANGLE_FAN, 0, 4)

        # Select the non-textured shader
        self.color.use()

        # Draw coastlines
        if self.wrapdir == 0:
            # Normal case, no wrap around
            self.coastlines.draw(gl.GL_LINES, 0, len(self.coastvertices), latlon=(0.0, 0.0))
        else:
            wrapindex = np.uint32(self.coastindices[int(self.wraplon)+180])
            if self.wrapdir == 1:
                self.coastlines.draw(gl.GL_LINES, 0, wrapindex, latlon=(0.0, 360.0))
                self.coastlines.draw(gl.GL_LINES, wrapindex, len(self.coastvertices) - wrapindex, latlon=(0.0, 0.0))
            else:
                self.coastlines.draw(gl.GL_LINES, 0, wrapindex, latlon=(0.0, 0.0))
                self.coastlines.draw(gl.GL_LINES, wrapindex, len(self.coastvertices) - wrapindex, latlon=(0.0, -360.0))

        # --- DRAW THE INSTANCED AIRCRAFT SHAPES ------------------------------
        # update wrap longitude and direction for the instanced objects
        BlueSkyProgram.enable_wrap(True)

        # PZ circles only when they are bigger than the A/C symbols
        if self.naircraft > 0 and self.show_traf and self.zoom >= 0.15:
            BlueSkyProgram.set_vertex_scale_type(VERTEX_IS_METERS)
            self.protectedzone.draw(gl.GL_LINE_STRIP, 0, self.vcount_circle, self.naircraft)

        BlueSkyProgram.set_vertex_scale_type(VERTEX_IS_SCREEN)

        # Draw traffic symbols
        if self.naircraft > 0 and self.show_traf:
            self.ac_symbol.draw(gl.GL_TRIANGLE_FAN, 0, 4, self.naircraft)

        # Draw waypoint symbols
        if self.show_wpt:
            self.waypoints.draw(gl.GL_LINE_LOOP, 0, 3, self.nwaypoints, color=self.color_wpt)

        # Draw airport symbols
        if self.show_apt:
            self.airports.draw(gl.GL_LINE_LOOP, 0, 4, self.nairports, color=self.color_apt)

        self.text.use()
        if self.show_apt:
            self.aptlabels.draw(color=self.color_aptlbl, n_instances=self.nairports)
        if self.zoom >= 0.5 and self.show_wpt:
            self.wptlabels.draw(color=self.color_wptlbl, n_instances=self.nwaypoints)
        if self.naircraft > 0 and self.show_traf and self.show_lbl:
            self.aclabels.draw(n_instances=self.naircraft)

        # Unbind everything
        RenderObject.unbind_all()
        gl.glUseProgram(0)

        self.swapBuffers()

        tend = time.time()
        #print("dt=%.4f [msec]\n"%((tend-tstart)*1000))

    def resizeGL(self, width, height):
        """Called upon window resizing: reinitialize the viewport."""
        # update the window size
        # Qt5 supports getting the device pixel ratio, which can be > 1 for HiDPI displays such as Mac Retina screens
        pixel_ratio = 1
        if QT_VERSION >= 5:
            pixel_ratio = self.devicePixelRatio()
        self.width, self.height = width / pixel_ratio, height / pixel_ratio

        self.ar = float(width) / float(height)
        BlueSkyProgram.set_aspect_ratio(self.ar)
        # paint within the whole window
        gl.glViewport(0, 0, width, height)

    def update_aircraft_data(self, data):
        n_ac = len(data.lat)
        if n_ac > 0:
            # Update data in GPU buffers
            update_array_buffer(self.aclatbuf, data.lat)
            update_array_buffer(self.aclonbuf, data.lon)
            update_array_buffer(self.achdgbuf, data.trk)
            # temp color
            color = np.zeros((n_ac, 3), dtype=np.float32)
            color[:,:] = (0.0, 1.0, 0.0)

            update_array_buffer(self.accolorbuf, color)

            # Labels
            rawlabel = ''
            for i in range(n_ac):
                rawlabel += '%-6sFL%03d %-6d' % (data.ids[i], int(data.alt[i] / ft / 100), int(data.tas[i] / kts))

            update_array_buffer(self.aclblbuf, np.array(rawlabel, dtype=np.string_))

        self.naircraft = n_ac

    def load_data(self):
        """Load static data: coaslines, waypoint and airport database.
        """
        # -------------------------COASTLINE DATA----------------------------------
        # Init geo (coastline)  data and convert pen up/pen down format of
        # coastlines to numpy arrays with lat/lon
        coast = []
        clat = clon = 0.0
        with open("data/global/coastlines.dat", 'r') as f:
            for line in f:
                if not (line.strip() == "" or line.strip()[0] == '#'):
                    arg = line.split()
                    if len(arg) == 3:
                        lat, lon = float(arg[1]), float(arg[2])
                        if arg[0] == 'D':
                            coast.append([clon, clat, lon, lat])
                        clat, clon = lat, lon
        # Sort the line segments by longitude of the first vertex
        self.coastvertices = np.array(
            sorted(coast, key=lambda a_entry: a_entry[0]), dtype=np.float32)
        self.coastindices = np.zeros(361)
        coastlon = self.coastvertices[:, 0]
        for i in range(0, 360):
            self.coastindices[i] = np.searchsorted(coastlon, i - 180) * 2
        self.coastvertices = self.coastvertices.reshape((self.coastvertices.size/2, 2))
        del coast

        # Waypoints data file. Example line:
        # ABARA, , 61.1833, 50.85, UUYY, High and Low Level, RS
        #  [id]    [lat]    [lon]  [airport]  [type] [country code]
        #   0  1     2       3         4        5         6
        wptlat = []
        wptlon = []
        wptids = ''
        with open('data/global/waypoints.dat', 'r') as f:
            for line in f:
                if not (line.strip() == "" or line.strip()[0] == '#'):
                    arg = line.split(',')
                    wptids += arg[0].ljust(5)
                    wptlat.append(float(arg[2]))
                    wptlon.append(float(arg[3]))
        self.wptlat = np.array(wptlat, dtype=np.float32)
        self.wptlon = np.array(wptlon, dtype=np.float32)
        self.wptlabeldata = np.array(wptids, dtype=np.string_)
        self.nwaypoints = len(self.wptlat)
        del wptlat, wptlon, wptids

        # Airports data file.
        # [code], [name], [lat], [lon], [class], [alt], [country code]
        #    0       1      2      3       4       5          6
        aptlat = []
        aptlon = []
        aptids = ''
        with open('data/global/airports.dat', 'r') as f:
            for line in f:
                if not (line.strip() == "" or line.strip()[0] == '#'):
                    arg = line.split(',')
                    # Skip apts without ID
                    if not arg[0].strip() == "":
                        aptids += arg[0].ljust(4)
                        aptlat.append(float(arg[2]))
                        aptlon.append(float(arg[3]))
        self.aptlat = np.array(aptlat, dtype=np.float32)
        self.aptlon = np.array(aptlon, dtype=np.float32)
        self.aptlabeldata = np.array(aptids, dtype=np.string_)
        self.nairports = len(self.aptlat)
        del aptlat, aptlon, aptids

    def pixelCoordsToGLxy(self, x, y):
        """Convert screen pixel coordinates to GL projection coordinates (x, y range -1 -- 1)
        """
        # GL coordinates (x, y range -1 -- 1)
        glx = (float(2.0 * x) / self.width  - 1.0)
        gly = -(float(2.0 * y) / self.height - 1.0)

        return (glx, gly)

    def pixelCoordsToLatLon(self, x, y):
        """Convert screen pixel coordinates to lat/lon coordinates
        """
        # GL coordinates (x, y range -1 -- 1)
        glx = (float(2.0 * x) / self.width  - 1.0)
        gly = -(float(2.0 * y) / self.height - 1.0)

        # glxy   = zoom * (pan + latlon)
        # latlon = glxy / zoom - pan
        lat = gly / (self.zoom * self.flat_earth) - self.pan[0]
        lon = glx / (self.zoom * self.ar) - self.pan[1]
        return (lat, lon)

    def event(self, event):
        if event.type() == PanZoomEvent.PanZoomEventType:
            if event.panzoom_type() == PanZoomEvent.Pan or event.panzoom_type() == PanZoomEvent.PanAbsolute:
                # Relative pan operation
                if event.panzoom_type() == PanZoomEvent.Pan:
                    self.pan[1] += event.pan()[1] / (self.zoom * self.flat_earth)
                    self.pan[0] += event.pan()[0] / (self.zoom * self.ar)
                # Absolute pan operation
                else:
                    self.pan[1] = event.pan()[1]
                    self.pan[0] = event.pan()[0]

                # Don't pan further than the poles in y-direction
                self.pan[0] = min(max(self.pan[0], -90.0 + 1.0 / (self.zoom * self.ar)), 90.0 - 1.0 / (self.zoom * self.ar))

                # Update flat-earth factor and possibly zoom in case of very wide windows (> 2:1)
                self.flat_earth = np.cos(np.deg2rad(self.pan[0]))
                self.zoom = max(self.zoom, 1.0 / (180.0 * self.flat_earth))

            elif event.panzoom_type() == PanZoomEvent.Zoom:
                prevzoom = self.zoom
                glxy = self.pixelCoordsToGLxy(event.origin()[0], event.origin()[1])
                self.zoom = event.zoom()

                # Limit zoom extents in x-direction to [-180:180], and in y-direction to [-90:90]
                self.zoom = max(self.zoom, 1.0 / min(90.0 * self.ar, 180.0 * self.flat_earth))

                # Correct pan so that zoom actions are around the mouse position, not around 0, 0
                # glxy / zoom1 - pan1 = glxy / zoom2 - pan2
                # pan2 = pan1 + glxy (1/zoom2 - 1/zoom1)
                self.pan[1] = self.pan[1] - glxy[0] * (1.0 / self.zoom - 1.0 / prevzoom) / self.flat_earth
                self.pan[0] = self.pan[0] - glxy[1] * (1.0 / self.zoom - 1.0 / prevzoom) / self.ar

                # Don't pan further than the poles in y-direction
                self.pan[0] = min(max(self.pan[0], -90.0 + 1.0 / (self.zoom * self.ar)), 90.0 - 1.0 / (self.zoom * self.ar))

                # Update flat-earth factor
                self.flat_earth = np.cos(np.deg2rad(self.pan[0]))
            event.accept()

            # Check for necessity wrap-around in x-direction
            self.wraplon  = -999.9
            self.wrapdir  = 0
            if self.pan[1] + 1.0 / (self.zoom * self.flat_earth) < -180.0:
                # The left edge of the map has passed the right edge of the screen: we can just change the pan position
                self.pan[1] += 360.0
            elif self.pan[1] - 1.0 / (self.zoom * self.flat_earth) < -180.0:
                # The left edge of the map has passed the left edge of the screen: we need to wrap around to the left
                self.wraplon = float(np.ceil(360.0 + self.pan[1] - 1.0 / (self.zoom * self.flat_earth)))
                self.wrapdir = -1
            elif self.pan[1] - 1.0 / (self.zoom * self.flat_earth) > 180.0:
                # The right edge of the map has passed the left edge of the screen: we can just change the pan position
                self.pan[1] -= 360.0
            elif self.pan[1] + 1.0 / (self.zoom * self.flat_earth) > 180.0:
                # The right edge of the map has passed the right edge of the screen: we need to wrap around to the right
                self.wraplon = float(np.floor(-360.0 + self.pan[1] + 1.0 / (self.zoom * self.flat_earth)))
                self.wrapdir = 1

            BlueSkyProgram.set_wrap(self.wraplon, self.wrapdir)

            # update pan and zoom on GPU for all shaders
            BlueSkyProgram.set_pan_and_zoom(self.pan[0], self.pan[1], self.zoom)

            return True

        else:
            return super(RadarWidget, self).event(event)
