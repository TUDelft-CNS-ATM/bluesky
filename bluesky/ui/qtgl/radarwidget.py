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

# Local imports
from ...tools.aero import ft, nm, kts
from glhelpers import BlueSkyProgram, RenderObject, TextObject, update_array_buffer
from uievents import PanZoomEvent, PanZoomEventType
from ...settings import text_size, apt_size, wpt_size, ac_size, font_family, font_weight, text_texture_size


VERTEX_IS_LATLON, VERTEX_IS_METERS, VERTEX_IS_SCREEN = range(3)

# Static defines
MAX_NAIRCRAFT    = 10000
MAX_ROUTE_LENGTH = 100


class RadarWidget(QGLWidget):
    show_traf = show_lbl = show_wpt = show_apt = True
    vcount_circle = 36
    width = height = 600
    panlat = panlon = 0.0
    zoom = 1.0
    ar = 1.0
    flat_earth = 1.0
    wraplon = int(-999)
    wrapdir = int(0)
    color_wpt = color_apt = (149.0/255.0, 179.0/255.0, 235/255.0)
    color_wptlbl = color_aptlbl = (219.0/255.0, 249.0/255.0, 255/255.0)
    color_route = (1.0, 0.0, 1.0)
    coastlinecolor = (84.0/255.0, 84.0/255.0, 114.0/255.0)

    def __init__(self, navdb):
        f = QGLFormat()
        f.setVersion(3, 3)
        f.setProfile(QGLFormat.CoreProfile)
        f.setDoubleBuffer(True)
        if QT_VERSION == 4:
            QGLWidget.__init__(self, QGLContext(f, None))
        else:
            # Qt 5
            QGLWidget.__init__(self, QGLContext(f))

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
        self.navdb = navdb

    def create_objects(self):
        # Initialize font for radar view with specified settings
        TextObject.create_font_array(char_height=text_texture_size, font_family=font_family, font_weight=font_weight)
        # Load and bind world texture
        # self.map_texture = load_texture('data/world.jpg')
        self.map_texture = self.bindTexture('data/graphics/world.16384x8192-mipmap.dds')

        # Create initial empty buffers for aircraft position, orientation, label, and color
        self.achdgbuf = RenderObject.create_empty_buffer(MAX_NAIRCRAFT * 4, usage=gl.GL_STREAM_DRAW)
        self.aclatbuf = RenderObject.create_empty_buffer(MAX_NAIRCRAFT * 4, usage=gl.GL_STREAM_DRAW)
        self.aclonbuf = RenderObject.create_empty_buffer(MAX_NAIRCRAFT * 4, usage=gl.GL_STREAM_DRAW)
        self.accolorbuf = RenderObject.create_empty_buffer(MAX_NAIRCRAFT * 12, usage=gl.GL_STREAM_DRAW)
        self.aclblbuf = RenderObject.create_empty_buffer(MAX_NAIRCRAFT * 15, usage=gl.GL_STREAM_DRAW)
        self.routebuf = RenderObject.create_empty_buffer(MAX_ROUTE_LENGTH * 8, usage=gl.GL_DYNAMIC_DRAW)

        # ------- Map ------------------------------------
        self.map = RenderObject()
        mapvertices = np.array([(540.0, -90.0), (-540.0, -90.0), (-540.0, 90.0), (540.0, 90.0)], dtype=np.float32)
        texcoords = np.array([(3, 1), (0, 1), (0, 0), (3, 0)], dtype=np.float32)
        self.map.bind_vertex_attribute(mapvertices)
        self.map.bind_texcoords_attribute(texcoords)

        # ------- Coastlines -----------------------------
        self.coastlines = RenderObject()
        coastvertices, coastindices = load_coast_data()
        self.coastlines.bind_vertex_attribute(coastvertices)
        self.coastlines.bind_color_attribute(np.array(self.coastlinecolor, dtype=np.float32))
        self.vcount_coast = len(coastvertices)
        self.coastindices = coastindices
        del coastvertices

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
        acvertices = np.array([(0.0, 0.5 * ac_size), (-0.5 * ac_size, -0.5 * ac_size), (0.0, -0.25 * ac_size), (0.5 * ac_size, -0.5 * ac_size)], dtype=np.float32)
        self.ac_symbol.bind_vertex_attribute(acvertices)
        self.ac_symbol.bind_lat_attribute(self.aclatbuf)
        self.ac_symbol.bind_lon_attribute(self.aclonbuf)
        self.ac_symbol.bind_orientation_attribute(self.achdgbuf)
        self.ac_symbol.bind_color_attribute(self.accolorbuf)
        self.aclabels = TextObject()
        self.aclabels.prepare_text_instanced(self.aclblbuf, self.aclatbuf, self.aclonbuf, (6, 3), self.accolorbuf, text_size=text_size, vertex_offset=(ac_size, -0.5 * ac_size))

        # ------- Aircraft Route -------------------------
        self.route = RenderObject()
        self.route.bind_vertex_attribute(self.routebuf)
        self.route.bind_color_attribute(np.array(self.color_route, dtype=np.float32))
        self.n_route_segments = 0

        # ------- Waypoints ------------------------------
        self.waypoints = RenderObject()
        wptvertices = np.array([(0.0, 0.5 * wpt_size), (-0.5 * wpt_size, -0.5 * wpt_size), (0.5 * wpt_size, -0.5 * wpt_size)], dtype=np.float32)  # a triangle
        self.waypoints.bind_vertex_attribute(wptvertices)
        self.nwaypoints = len(self.navdb.wplat)
        self.wptlatbuf = self.waypoints.bind_lat_attribute(np.array(self.navdb.wplat, dtype=np.float32))
        self.wptlonbuf = self.waypoints.bind_lon_attribute(np.array(self.navdb.wplon, dtype=np.float32))
        self.wptlabels = TextObject()
        wptids = ''
        for wptid in self.navdb.wpid:
            wptids += wptid.ljust(5)
        self.wptlabels.prepare_text_instanced(np.array(wptids, dtype=np.string_), self.wptlatbuf, self.wptlonbuf, (5, 1), text_size=text_size, vertex_offset=(wpt_size, 0.5 * wpt_size))
        del wptids

        # ------- Airports -------------------------------
        self.airports = RenderObject()
        aptvertices = np.array([(-0.5 * apt_size, -0.5 * apt_size), (0.5 * apt_size, -0.5 * apt_size), (0.5 * apt_size, 0.5 * apt_size), (-0.5 * apt_size, 0.5 * apt_size)], dtype=np.float32)  # a square
        self.airports.bind_vertex_attribute(aptvertices)
        self.nairports = len(self.navdb.aplat)
        self.aptlatbuf = self.airports.bind_lat_attribute(np.array(self.navdb.aplat, dtype=np.float32))
        self.aptlonbuf = self.airports.bind_lon_attribute(np.array(self.navdb.aplon, dtype=np.float32))
        self.aptlabels = TextObject()
        aptids = ''
        for aptid in self.navdb.apid:
            aptids += aptid.ljust(4)
        self.aptlabels.prepare_text_instanced(np.array(aptids, dtype=np.string_), self.aptlatbuf, self.aptlonbuf, (4, 1), text_size=text_size, vertex_offset=(apt_size, 0.5 * apt_size))
        del aptids

        # Unbind VAO, VBO
        RenderObject.unbind_all()

        # Set initial values for the global uniforms
        BlueSkyProgram.set_wrap(self.wraplon, self.wrapdir)
        BlueSkyProgram.set_pan_and_zoom(self.panlat, self.panlon, self.zoom)

    def initializeGL(self):
        """Initialize OpenGL, VBOs, upload data on the GPU, etc."""

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
            self.coastlines.draw(gl.GL_LINES, 0, self.vcount_coast, latlon=(0.0, 0.0))
        else:
            wrapindex = np.uint32(self.coastindices[int(self.wraplon)+180])
            if self.wrapdir == 1:
                self.coastlines.draw(gl.GL_LINES, 0, wrapindex, latlon=(0.0, 360.0))
                self.coastlines.draw(gl.GL_LINES, wrapindex, self.vcount_coast - wrapindex, latlon=(0.0, 0.0))
            else:
                self.coastlines.draw(gl.GL_LINES, 0, wrapindex, latlon=(0.0, 0.0))
                self.coastlines.draw(gl.GL_LINES, wrapindex, self.vcount_coast - wrapindex, latlon=(0.0, -360.0))

        # --- DRAW THE SELECTED AIRCRAFT ROUTE (WHEN AVAILABLE) ---------------
        if self.n_route_segments > 0:
            self.route.draw(gl.GL_LINE_STRIP, 0, self.n_route_segments)

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
        self.ar = float(width) / float(height)
        BlueSkyProgram.set_win_width_height(width, height)

        # paint within the whole window
        gl.glViewport(0, 0, width, height)

        # Update zoom
        self.event(PanZoomEvent(PanZoomEvent.Zoom, zoom, origin))

    def update_route_data(self, data):
        self.route_acidx      = data.acidx
        if self.route_acidx >= 0:
            self.n_route_segments = len(data.lat)
            routedata       = np.empty((self.n_route_segments, 2), dtype=np.float32)
            routedata[:, 0] = data.lon
            routedata[:, 1] = data.lat
            routedata.resize((1, 2 * self.n_route_segments))

            update_array_buffer(self.routebuf, routedata)
        else:
            self.n_route_segments = 0

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
                rawlabel += '%-6sFL%03d %-6d' % (data.id[i], int(data.alt[i] / ft / 100), int(data.tas[i] / kts))

            update_array_buffer(self.aclblbuf, np.array(rawlabel, dtype=np.string_))

        self.naircraft = n_ac

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
        return (lat, lon)

    def event(self, event):
        if event.type() == PanZoomEventType:
            if event.panzoom_type() == PanZoomEvent.Pan or event.panzoom_type() == PanZoomEvent.PanAbsolute:
                # Relative pan operation
                if event.panzoom_type() == PanZoomEvent.Pan:
                    self.panlat += event.panlat() / (self.zoom * self.ar)
                    self.panlon += event.panlon() / (self.zoom * self.flat_earth)
                # Absolute pan operation
                else:
                    self.panlat = event.panlat()
                    self.panlon = event.panlon()

                # Don't pan further than the poles in y-direction
                self.panlat = min(max(self.panlat, -90.0 + 1.0 / (self.zoom * self.ar)), 90.0 - 1.0 / (self.zoom * self.ar))

                # Update flat-earth factor and possibly zoom in case of very wide windows (> 2:1)
                self.flat_earth = np.cos(np.deg2rad(self.panlat))
                self.zoom = max(self.zoom, 1.0 / (180.0 * self.flat_earth))

            elif event.panzoom_type() == PanZoomEvent.Zoom:
                prevzoom = self.zoom
                glx, gly = self.pixelCoordsToGLxy(event.origin()[0], event.origin()[1])
                self.zoom *= event.zoom()

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

            BlueSkyProgram.set_wrap(self.wraplon, self.wrapdir)

            # update pan and zoom on GPU for all shaders
            BlueSkyProgram.set_pan_and_zoom(self.panlat, self.panlon, self.zoom)

            return True

        else:
            return super(RadarWidget, self).event(event)


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
            if not (line.strip() == "" or line.strip()[0] == '#'):
                arg = line.split()
                if len(arg) == 3:
                    lat, lon = float(arg[1]), float(arg[2])
                    if arg[0] == 'D':
                        coast.append([clon, clat, lon, lat])
                    clat, clon = lat, lon
    # Sort the line segments by longitude of the first vertex
    coastvertices = np.array(
        sorted(coast, key=lambda a_entry: a_entry[0]), dtype=np.float32)
    coastindices = np.zeros(361)
    coastlon = coastvertices[:, 0]
    for i in range(0, 360):
        coastindices[i] = np.searchsorted(coastlon, i - 180) * 2
    coastvertices = coastvertices.reshape((coastvertices.size/2, 2))
    del coast
    return coastvertices, coastindices
