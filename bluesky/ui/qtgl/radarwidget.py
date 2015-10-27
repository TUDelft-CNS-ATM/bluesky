try:
    from PyQt4.QtCore import Qt, qCritical
    from PyQt4.QtOpenGL import QGLWidget
    QT_VERSION = 4
except ImportError:
    from PyQt5.QtCore import Qt, qCritical
    from PyQt5.QtOpenGL import QGLWidget
    QT_VERSION = 5
import numpy as np
import OpenGL.GL as gl
from ctypes import c_float, c_int, Structure

# Local imports
from ...tools.aero import ft, nm, kts
from glhelpers import BlueSkyProgram, RenderObject, TextObject, update_array_buffer, UniformBuffer
from uievents import PanZoomEvent, PanZoomEventType
from ...settings import text_size, apt_size, wpt_size, ac_size, font_family, font_weight, text_texture_size


# Static defines
MAX_NAIRCRAFT        = 10000
MAX_NCONFLICTS       = 1000
MAX_ROUTE_LENGTH     = 100
MAX_POLYGON_SEGMENTS = 100

# Colors
red   = (1.0, 0.0, 0.0)
green = (0.0, 1.0, 0.0)
blue  = (0.0, 0.0, 1.0)
lightblue = (0.0, 0.8, 1.0)
cyan  = (0.0, 1.0, 0.0)
amber = (1.0, 0.6, 0.0)

VERTEX_IS_LATLON, VERTEX_IS_METERS, VERTEX_IS_SCREEN = range(3)


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
    panlat = 51.5
    panlon = 6.5
    zoom = 0.5
    ar = 1.0
    flat_earth = 1.0
    wraplon = int(-999)
    wrapdir = int(0)
    color_wpt = color_apt = (149.0/255.0, 179.0/255.0, 235/255.0)
    color_wptlbl = color_aptlbl = (219.0/255.0, 249.0/255.0, 255/255.0)
    color_route = (1.0, 0.0, 1.0)
    coastlinecolor = (84.0/255.0, 84.0/255.0, 114.0/255.0)

    do_text = True

    def __init__(self, navdb, shareWidget=None):
        super(RadarWidget, self).__init__(shareWidget=shareWidget)
        self.setAttribute(Qt.WA_AcceptTouchEvents, True)
        self.grabGesture(Qt.PanGesture)
        self.grabGesture(Qt.PinchGesture)
        self.grabGesture(Qt.SwipeGesture)

        # The number of aircraft in the simulation
        self.naircraft   = 0
        self.nwaypoints  = 0
        self.nairports   = 0
        self.route_acidx = -1
        self.navdb       = navdb

        self.initialized = False

    def create_objects(self):
        # Initialize font for radar view with specified settings
        TextObject.create_font_array(char_height=text_texture_size, font_family=font_family, font_weight=font_weight)
        # Load and bind world texture
        # self.map_texture = load_texture('data/world.jpg')
        self.map_texture = self.bindTexture('data/graphics/world.16384x8192-mipmap.dds')

        # Create initial empty buffers for aircraft position, orientation, label, and color
        self.achdgbuf    = RenderObject.create_empty_buffer(MAX_NAIRCRAFT * 4, usage=gl.GL_STREAM_DRAW)
        self.aclatbuf    = RenderObject.create_empty_buffer(MAX_NAIRCRAFT * 4, usage=gl.GL_STREAM_DRAW)
        self.aclonbuf    = RenderObject.create_empty_buffer(MAX_NAIRCRAFT * 4, usage=gl.GL_STREAM_DRAW)
        self.accolorbuf  = RenderObject.create_empty_buffer(MAX_NAIRCRAFT * 12, usage=gl.GL_STREAM_DRAW)
        self.aclblbuf    = RenderObject.create_empty_buffer(MAX_NAIRCRAFT * 15, usage=gl.GL_STREAM_DRAW)
        self.confcpabuf  = RenderObject.create_empty_buffer(MAX_NCONFLICTS * 8, usage=gl.GL_STREAM_DRAW)
        self.polyprevbuf = RenderObject.create_empty_buffer(MAX_POLYGON_SEGMENTS * 8, usage=gl.GL_DYNAMIC_DRAW)
        self.routebuf    = RenderObject.create_empty_buffer(MAX_ROUTE_LENGTH * 8, usage=gl.GL_DYNAMIC_DRAW)

        # ------- Map ------------------------------------
        self.map = RenderObject(gl.GL_TRIANGLE_FAN, vertex_count=4)
        mapvertices = np.array([(-90.0, 540.0), (-90.0, -540.0), (90.0, -540.0), (90.0, 540.0)], dtype=np.float32)
        texcoords = np.array([(1, 3), (1, 0), (0, 0), (0, 3)], dtype=np.float32)
        self.map.bind_vertex_attribute(mapvertices)
        self.map.bind_texcoords_attribute(texcoords)

        # ------- Coastlines -----------------------------
        self.coastlines = RenderObject(gl.GL_LINES)
        coastvertices, coastindices = load_coast_data()
        self.coastlines.bind_vertex_attribute(coastvertices)
        self.coastlines.bind_color_attribute(np.array(self.coastlinecolor, dtype=np.float32))
        self.vcount_coast = len(coastvertices)
        self.coastindices = coastindices
        del coastvertices

        # Polygon preview object
        self.polyprev = RenderObject(gl.GL_LINE_LOOP)
        self.polyprev.bind_vertex_attribute(self.polyprevbuf)
        self.polyprev.bind_color_attribute(np.array(lightblue, dtype=np.float32))

        # ------- Circle ---------------------------------
        # Create a new VAO (Vertex Array Object) and bind it
        self.protectedzone = RenderObject(gl.GL_LINE_LOOP, vertex_count=self.vcount_circle)
        circlevertices = np.transpose(np.array((5.0*nm*np.cos(np.linspace(0.0, 2.0*np.pi, self.vcount_circle)), 5.0*nm*np.sin(np.linspace(0.0, 2.0*np.pi, self.vcount_circle))), dtype=np.float32))
        self.protectedzone.bind_vertex_attribute(circlevertices)
        self.protectedzone.bind_lat_attribute(self.aclatbuf)
        self.protectedzone.bind_lon_attribute(self.aclonbuf)
        self.protectedzone.bind_color_attribute(self.accolorbuf)

        # ------- A/C symbol -----------------------------
        self.ac_symbol = RenderObject(gl.GL_TRIANGLE_FAN, vertex_count=4)
        acvertices = np.array([(0.0, 0.5 * ac_size), (-0.5 * ac_size, -0.5 * ac_size), (0.0, -0.25 * ac_size), (0.5 * ac_size, -0.5 * ac_size)], dtype=np.float32)
        self.ac_symbol.bind_vertex_attribute(acvertices)
        self.ac_symbol.bind_lat_attribute(self.aclatbuf)
        self.ac_symbol.bind_lon_attribute(self.aclonbuf)
        self.ac_symbol.bind_orientation_attribute(self.achdgbuf)
        self.ac_symbol.bind_color_attribute(self.accolorbuf)
        self.aclabels = TextObject()
        self.aclabels.prepare_text_instanced(self.aclblbuf, self.aclatbuf, self.aclonbuf, (6, 3), self.accolorbuf, text_size=text_size, vertex_offset=(ac_size, -0.5 * ac_size))

        # ------- Conflict CPA lines ---------------------
        self.cpalines = RenderObject(gl.GL_LINES)
        self.cpalines.bind_vertex_attribute(self.confcpabuf)
        self.cpalines.bind_color_attribute(np.array(amber, dtype=np.float32))

        # ------- Aircraft Route -------------------------
        self.route = RenderObject(gl.GL_LINE_STRIP)
        self.route.bind_vertex_attribute(self.routebuf)
        self.route.bind_color_attribute(np.array(self.color_route, dtype=np.float32))

        # ------- Waypoints ------------------------------
        self.nwaypoints = len(self.navdb.wplat)
        self.waypoints = RenderObject(gl.GL_LINE_LOOP, vertex_count=3, n_instances=self.nwaypoints)
        wptvertices = np.array([(0.0, 0.5 * wpt_size), (-0.5 * wpt_size, -0.5 * wpt_size), (0.5 * wpt_size, -0.5 * wpt_size)], dtype=np.float32)  # a triangle
        self.waypoints.bind_vertex_attribute(wptvertices)
        self.wptlatbuf = self.waypoints.bind_lat_attribute(np.array(self.navdb.wplat, dtype=np.float32))
        self.wptlonbuf = self.waypoints.bind_lon_attribute(np.array(self.navdb.wplon, dtype=np.float32))
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
        self.airports.bind_vertex_attribute(aptvertices)
        indices = self.navdb.aptype.argsort()
        aplat   = np.array(self.navdb.aplat[indices], dtype=np.float32)
        aplon   = np.array(self.navdb.aplon[indices], dtype=np.float32)
        aptypes = self.navdb.aptype[indices]
        apnames = np.array(self.navdb.apid)
        apnames = apnames[indices]
        # The number of large, large+med, and large+med+small airports
        self.nairports = [aptypes.searchsorted(2), aptypes.searchsorted(3), self.nairports]

        self.aptlatbuf = self.airports.bind_lat_attribute(aplat)
        self.aptlonbuf = self.airports.bind_lon_attribute(aplon)
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

    def initializeGL(self):
        """Initialize OpenGL, VBOs, upload data on the GPU, etc."""

        # background color
        gl.glClearColor(0, 0, 0, 0)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        self.globaldata = radarUBO()

        try:
            # Compile shaders and link color shader program
            self.color = BlueSkyProgram('data/graphics/shaders/radarwidget-normal.vert', 'data/graphics/shaders/radarwidget-color.frag')
            self.color.bind_uniform_buffer('global_data', self.globaldata)

            # Compile shaders and link texture shader program
            self.texture = BlueSkyProgram('data/graphics/shaders/radarwidget-normal.vert', 'data/graphics/shaders/radarwidget-texture.frag')
            self.texture.bind_uniform_buffer('global_data', self.globaldata)

            # Compile shaders and link text shader program
            self.text = BlueSkyProgram('data/graphics/shaders/radarwidget-text.vert', 'data/graphics/shaders/radarwidget-text.frag')
            self.text.bind_uniform_buffer('global_data', self.globaldata)
            TextObject.init_shader(self.text)
        except RuntimeError as e:
            qCritical('Error compiling shaders in radarwidget: ' + e.args[0])
            return

        # create a vertex array objects
        self.create_objects()

        # Done initializing
        self.initialized = True

    def paintGL(self):
        """Paint the scene."""
        # pass if the framebuffer isn't complete yet or if not initialized
        if not (gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) == gl.GL_FRAMEBUFFER_COMPLETE and self.initialized and self.isVisible()):
            return

        # clear the buffer
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        # Send the (possibly) updated global uniforms to the buffer
        self.globaldata.set_vertex_scale_type(VERTEX_IS_LATLON)

        # --- DRAW THE MAP AND COASTLINES ---------------------------------------------
        # Map and coastlines: don't wrap around in the shader
        self.globaldata.enable_wrap(False)

        if self.show_map:
            # Select the texture shader
            self.texture.use()

            # Draw map texture
            gl.glActiveTexture(gl.GL_TEXTURE0 + 0)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.map_texture)
            self.map.draw()

        # Select the non-textured shader
        self.color.use()

        # Draw coastlines
        if self.show_coast:
            if self.wrapdir == 0:
                # Normal case, no wrap around
                self.coastlines.draw(first_vertex=0, vertex_count=self.vcount_coast, latlon=(0.0, 0.0))
            else:
                wrapindex = np.uint32(self.coastindices[int(self.wraplon)+180])
                if self.wrapdir == 1:
                    self.coastlines.draw(first_vertex=0, vertex_count=wrapindex, latlon=(0.0, 360.0))
                    self.coastlines.draw(first_vertex=wrapindex, vertex_count=self.vcount_coast - wrapindex, latlon=(0.0, 0.0))
                else:
                    self.coastlines.draw(first_vertex=0, vertex_count=wrapindex, latlon=(0.0, 0.0))
                    self.coastlines.draw(first_vertex=wrapindex, vertex_count=self.vcount_coast - wrapindex, latlon=(0.0, -360.0))

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
            self.waypoints.draw(n_instances=self.nwaypoints, color=self.color_wpt)

        # Draw airport symbols
        if self.show_apt:
            self.airports.draw(n_instances=nairports, color=self.color_apt)

        if self.do_text:
            self.text.use()

            if self.show_apt:
                self.aptlabels.draw(color=self.color_aptlbl, n_instances=nairports)
            if self.zoom >= 1.0 and show_wpt:
                self.wptlabels.draw(color=self.color_wptlbl, n_instances=self.nwaypoints)
            if self.naircraft > 0 and self.show_traf and self.show_lbl:
                self.aclabels.draw(n_instances=self.naircraft)

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

        # Update zoom
        self.event(PanZoomEvent(zoom=zoom, origin=origin))

    def update_route_data(self, data):
        self.route_acidx      = data.acidx
        if self.route_acidx >= 0:
            nsegments = len(data.lat)
            self.route.set_vertex_count(nsegments)
            routedata       = np.empty((nsegments, 2), dtype=np.float32)
            routedata[:, 0] = data.lon
            routedata[:, 1] = data.lat
            routedata.resize((1, 2 * nsegments))

            update_array_buffer(self.routebuf, routedata)
        else:
            self.route.set_vertex_count(0)

    def update_aircraft_data(self, data):
        self.naircraft = len(data.lat)
        if self.naircraft > 0:
            # Update data in GPU buffers
            update_array_buffer(self.aclatbuf, data.lat)
            update_array_buffer(self.aclonbuf, data.lon)
            update_array_buffer(self.achdgbuf, data.trk)

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

            update_array_buffer(self.confcpabuf, cpalines)
            update_array_buffer(self.accolorbuf, color)
            update_array_buffer(self.aclblbuf, np.array(rawlabel, dtype=np.string_))

            # If there is a visible route, update the start position
            if self.route_acidx >= 0:
                update_array_buffer(self.routebuf, np.array([data.lon[self.route_acidx], data.lat[self.route_acidx]], dtype=np.float32))

    def updatePolygon(self, name, data_in):
        self.makeCurrent()
        if name in self.polys:
            if data_in is None:
                del self.polys[name]
            else:
                update_array_buffer(self.polys[name].vertexbuf, data_in)
                self.polys[name].set_vertex_count(len(data_in) / 2)
        else:
            if data_in is None:
                print "Delete '" + name + "': object not found!"
            else:
                newpoly = RenderObject(gl.GL_LINE_LOOP, vertex_count=len(data_in)/2)
                newpoly.bind_vertex_attribute(data_in)
                newpoly.bind_color_attribute(np.array(blue, dtype=np.float32))
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
        update_array_buffer(self.polyprevbuf, data)
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
                self.panlat = min(max(self.panlat, -90.0 + 1.0 /   \
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
                        coast.append([clat, clon, lat, lon])
                    clat, clon = lat, lon
    # Sort the line segments by longitude of the first vertex
    coastvertices = np.array(
        sorted(coast, key=lambda a_entry: a_entry[0]), dtype=np.float32)
    coastindices = np.zeros(361)
    coastlon = coastvertices[:, 0]
    for i in range(0, 360):
        coastindices[i] = np.searchsorted(coastlon, i - 180) * 2
    coastvertices.resize((coastvertices.size/2, 2))
    del coast
    return coastvertices, coastindices
