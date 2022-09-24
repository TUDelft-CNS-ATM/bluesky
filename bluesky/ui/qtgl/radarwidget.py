''' BlueSky OpenGL radar view. '''
from ctypes import c_float, c_int, Structure
import numpy as np

try:
    from PyQt5.QtCore import Qt, QEvent, QT_VERSION
except ImportError:
    from PyQt6.QtCore import Qt, QEvent, QT_VERSION

import bluesky as bs
from bluesky.core import Signal
from bluesky.ui.qtgl import glhelpers as glh
from bluesky.ui.radarclick import radarclick
from bluesky.ui.qtgl import console
from .gltraffic import Traffic
from .glmap import Map
from .glnavdata import Navdata
from .glpoly import Poly
from .gltiledmap import TiledMap
# Register settings defaults
bs.settings.set_variable_defaults(gfx_path='graphics')

# Qt smaller than 5.6.2 needs a different approach to pinch gestures
CORRECT_PINCH = False
if QT_VERSION <= 0x050600:
    import platform
    CORRECT_PINCH = platform.system() == 'Darwin'


class RadarShaders(glh.ShaderSet):
    ''' Shaderset for the radar view. '''
    # Vertex type enum: Individual vertices can correspond to lat/lon coordinates, screen coordinates, or meters.
    VERTEX_IS_LATLON, VERTEX_IS_METERS, VERTEX_IS_SCREEN = list(range(3))
    def __init__(self, parent):
        super().__init__(parent)

        class GlobalData(Structure):
            _fields_ = [("wrapdir", c_int), ("wraplon", c_float), ("panlat", c_float), ("panlon", c_float),
                        ("zoom", c_float), ("screen_width", c_int), ("screen_height", c_int), ("vertex_scale_type", c_int),
                        ("screen_pixel_ratio", c_float)]
        self.data = GlobalData()

    def create(self):
        super().create()
        shaderpath = (bs.resource(bs.settings.gfx_path) / 'shaders').as_posix()
        self.set_shader_path(shaderpath)
        # Load all shaders for this shader set
        self.load_shader('normal', 'radarwidget-normal.vert',
                         'radarwidget-color.frag')
        self.load_shader('textured', 'radarwidget-normal.vert',
                         'radarwidget-texture.frag')
        self.load_shader('tiled', 'radarwidget-normal.vert',
                         'radarwidget-tiled.frag')
        self.load_shader('text', 'radarwidget-text.vert',
                         'radarwidget-text.frag')
        self.load_shader('ssd', 'ssd.vert', 'ssd.frag', 'ssd.geom')

    def set_wrap(self, wraplon, wrapdir):
        self.data.wrapdir = wrapdir
        self.data.wraplon = wraplon

    def set_pan_and_zoom(self, panlat, panlon, zoom):
        self.data.panlat = panlat
        self.data.panlon = panlon
        self.data.zoom = zoom

    def set_pixel_ratio(self, pxratio):
        self.data.screen_pixel_ratio = pxratio

    def set_win_width_height(self, w, h):
        self.data.screen_width = w
        self.data.screen_height = h

    def enable_wrap(self, flag=True):
        if not flag:
            wrapdir = self.data.wrapdir
            self.data.wrapdir = 0
            self.update_ubo('global_data', self.data, 0, 4)
            self.data.wrapdir = wrapdir
        else:
            self.update_ubo('global_data', self.data, 0, 4)

    def set_vertex_scale_type(self, vertex_scale_type):
        self.data.vertex_scale_type = vertex_scale_type
        self.update_ubo('global_data', self.data)


class RadarWidget(glh.RenderWidget):
    ''' The BlueSky radar view. '''
    def __init__(self, parent=None):
        super().__init__(parent)
        self.prevwidth = self.prevheight = 600
        self.pxratio = 1
        self.panlat = 0.0
        self.panlon = 0.0
        self.zoom = 1.0
        self.ar = 1.0
        self.flat_earth = 1.0
        self.wraplon = int(-999)
        self.wrapdir = int(0)
        self.initialized = False

        self.panzoomchanged = False
        self.mousedragged = False
        self.mousepos = (0, 0)
        self.prevmousepos = (0, 0)

        self.shaderset = RadarShaders(self)
        self.set_shaderset(self.shaderset)

        # Add default objects
        self.addobject(Map(parent=self))
        self.addobject(Traffic(parent=self))
        self.addobject(Navdata(parent=self))
        self.addobject(Poly(parent=self))

        self.setAttribute(Qt.WidgetAttribute.WA_AcceptTouchEvents, True)
        self.grabGesture(Qt.GestureType.PanGesture)
        self.grabGesture(Qt.GestureType.PinchGesture)
        # self.grabGesture(Qt.SwipeGesture)
        self.setMouseTracking(True)

        # Signals and slots
        bs.net.actnodedata_changed.connect(self.actdata_changed)
        self.mouse_event = Signal('radarmouse')
        self.panzoom_event = Signal('panzoom')

    def actdata_changed(self, nodeid, nodedata, changed_elems):
        ''' Update buffers when a different node is selected, or when
            the data of the current node is updated. '''

        # Update pan/zoom
        if 'PANZOOM' in changed_elems:
            self.panzoom(pan=nodedata.pan, zoom=nodedata.zoom, absolute=True)

    def initializeGL(self):
        """Initialize OpenGL, VBOs, upload data on the GPU, etc."""
        super().initializeGL()

        # Set initial values for the global uniforms
        self.shaderset.set_wrap(self.wraplon, self.wrapdir)
        self.shaderset.set_pan_and_zoom(self.panlat, self.panlon, self.zoom)

        # background color
        glh.gl.glClearColor(0.7, 0.7, 0.7, 0)
        glh.gl.glEnable(glh.gl.GL_BLEND)
        glh.gl.glBlendFunc(glh.gl.GL_SRC_ALPHA, glh.gl.GL_ONE_MINUS_SRC_ALPHA)

        self.initialized = True


    def resizeGL(self, width, height):
        """Called upon window resizing: reinitialize the viewport."""
        # update the window size

        # Calculate zoom so that the window resize doesn't affect the scale, but only enlarges or shrinks the view
        zoom = float(self.prevwidth) / float(width)
        origin = (width / 2, height / 2)

        # Update width, height, and aspect ratio
        self.prevwidth, self.prevheight = width, height
        self.ar = float(width) / max(1, float(height))
        self.pxratio = self.devicePixelRatio()
        self.shaderset.set_pixel_ratio(self.pxratio)
        self.shaderset.set_win_width_height(width, height)
        # Update zoom
        self.panzoom(zoom=zoom, origin=origin)

    def pixelCoordsToGLxy(self, x, y):
        """Convert screen pixel coordinates to GL projection coordinates (x, y range -1 -- 1)
        """
        # GL coordinates (x, y range -1 -- 1)
        glx = (float(2.0 * x) / self.prevwidth - 1.0)
        gly = -(float(2.0 * y) / self.prevheight - 1.0)
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

    def viewportlatlon(self):
        ''' Return the viewport bounds in lat/lon coordinates. '''
        return (self.panlat + 1.0 / (self.zoom * self.ar),
                self.panlon - 1.0 / (self.zoom * self.flat_earth),
                self.panlat - 1.0 / (self.zoom * self.ar),
                self.panlon + 1.0 / (self.zoom * self.flat_earth))

    def panzoom(self, pan=None, zoom=None, origin=None, absolute=False):
        if not self.initialized:
            return False

        if pan:
            # Absolute pan operation
            if absolute:
                self.panlat = pan[0]
                self.panlon = pan[1]
            # Relative pan operation
            else:
                self.panlat += pan[0]
                self.panlon += pan[1]

            # Don't pan further than the poles in y-direction
            self.panlat = min(max(self.panlat, -90.0 + 1.0 /
                                  (self.zoom * self.ar)), 90.0 - 1.0 / (self.zoom * self.ar))

            # Update flat-earth factor and possibly zoom in case of very wide windows (> 2:1)
            self.flat_earth = np.cos(np.deg2rad(self.panlat))
            self.zoom = max(self.zoom, 1.0 / (180.0 * self.flat_earth))

        if zoom:
            if absolute:
                # Limit zoom extents in x-direction to [-180:180], and in y-direction to [-90:90]
                self.zoom = max(
                    zoom, 1.0 / min(90.0 * self.ar, 180.0 * self.flat_earth))
            else:
                prevzoom = self.zoom
                glx, gly = self.pixelCoordsToGLxy(
                    *origin) if origin else (0, 0)
                self.zoom *= zoom

                # Limit zoom extents in x-direction to [-180:180], and in y-direction to [-90:90]
                self.zoom = max(self.zoom, 1.0 / min(90.0 *
                                                     self.ar, 180.0 * self.flat_earth))

                # Correct pan so that zoom actions are around the mouse position, not around 0, 0
                # glxy / zoom1 - pan1 = glxy / zoom2 - pan2
                # pan2 = pan1 + glxy (1/zoom2 - 1/zoom1)
                self.panlon = self.panlon - glx * \
                    (1.0 / self.zoom - 1.0 / prevzoom) / self.flat_earth
                self.panlat = self.panlat - gly * \
                    (1.0 / self.zoom - 1.0 / prevzoom) / self.ar

            # Don't pan further than the poles in y-direction
            self.panlat = min(max(self.panlat, -90.0 + 1.0 /
                                  (self.zoom * self.ar)), 90.0 - 1.0 / (self.zoom * self.ar))

            # Update flat-earth factor
            self.flat_earth = np.cos(np.deg2rad(self.panlat))

        # Check for necessity wrap-around in x-direction
        self.wraplon = -999.9
        self.wrapdir = 0
        if self.panlon + 1.0 / (self.zoom * self.flat_earth) < -180.0:
            # The left edge of the map has passed the right edge of the screen: we can just change the pan position
            self.panlon += 360.0
        elif self.panlon - 1.0 / (self.zoom * self.flat_earth) < -180.0:
            # The left edge of the map has passed the left edge of the screen: we need to wrap around to the left
            self.wraplon = float(
                np.ceil(360.0 + self.panlon - 1.0 / (self.zoom * self.flat_earth)))
            self.wrapdir = -1
        elif self.panlon - 1.0 / (self.zoom * self.flat_earth) > 180.0:
            # The right edge of the map has passed the left edge of the screen: we can just change the pan position
            self.panlon -= 360.0
        elif self.panlon + 1.0 / (self.zoom * self.flat_earth) > 180.0:
            # The right edge of the map has passed the right edge of the screen: we need to wrap around to the right
            self.wraplon = float(
                np.floor(-360.0 + self.panlon + 1.0 / (self.zoom * self.flat_earth)))
            self.wrapdir = 1

        self.shaderset.set_wrap(self.wraplon, self.wrapdir)

        # update pan and zoom on GPU for all shaders
        self.shaderset.set_pan_and_zoom(self.panlat, self.panlon, self.zoom)
        # Update pan and zoom in centralized nodedata
        bs.net.get_nodedata().panzoom((self.panlat, self.panlon), self.zoom)
        self.panzoom_event.emit(False)
        return True

    def event(self, event):
        ''' Event handling for input events. '''
        if event.type() == QEvent.Type.Wheel:
            # For mice we zoom with control/command and the scrolwheel
            if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                origin = (event.pos().x(), event.pos().y())
                zoom = 1.0
                try:
                    if event.pixelDelta():
                        # High resolution scroll
                        zoom *= (1.0 + 0.01 * event.pixelDelta().y())
                    else:
                        # Low resolution scroll
                        zoom *= (1.0 + 0.001 * event.angleDelta().y())
                except AttributeError:
                    zoom *= (1.0 + 0.001 * event.delta())
                self.panzoomchanged = True
                return self.panzoom(zoom=zoom, origin=origin)

            # For touchpad scroll (2D) is used for panning
            else:
                try:
                    dlat = 0.01 * event.pixelDelta().y() / (self.zoom * self.ar)
                    dlon = -0.01 * event.pixelDelta().x() / (self.zoom * self.flat_earth)
                    self.panzoomchanged = True
                    return self.panzoom(pan=(dlat, dlon))
                except AttributeError:
                    pass

        # For touchpad, pinch gesture is used for zoom
        elif event.type() == QEvent.Type.Gesture:
            pan = zoom = None
            dlat = dlon = 0.0
            for g in event.gestures():
                if g.gestureType() == Qt.GestureType.PinchGesture:
                    event.accept(g)
                    zoom = g.scaleFactor() * (zoom or 1.0)
                    if CORRECT_PINCH:
                        zoom /= g.lastScaleFactor()
                elif g.gestureType() == Qt.GestureType.PanGesture:
                    event.accept(g)
                    if abs(g.delta().y() + g.delta().x()) > 1e-1:
                        dlat += 0.005 * g.delta().y() / (self.zoom * self.ar)
                        dlon -= 0.005 * g.delta().x() / (self.zoom * self.flat_earth)
                        pan = (dlat, dlon)
            if pan is not None or zoom is not None:
                self.panzoomchanged = True
                return self.panzoom(pan, zoom, self.mousepos)

        elif event.type() == QEvent.Type.MouseButtonPress and event.button() & Qt.MouseButton.LeftButton:
            self.mousedragged = False
            # For mice we pan with control/command and mouse movement.
            # Mouse button press marks the beginning of a pan
            self.prevmousepos = (event.pos().x(), event.pos().y())

        elif event.type() == QEvent.Type.MouseButtonRelease and \
                event.button() & Qt.MouseButton.LeftButton and not self.mousedragged:
            lat, lon = self.pixelCoordsToLatLon(event.pos().x(), event.pos().y())
            actdata = bs.net.get_nodedata()
            tostack, tocmdline = radarclick(console.get_cmdline(), lat, lon,
                                            actdata.acdata, actdata.routedata)

            console.process_cmdline((tostack + '\n' + tocmdline) if tostack else tocmdline)

        elif event.type() == QEvent.Type.MouseMove:
            self.mousedragged = True
            self.mousepos = (event.pos().x(), event.pos().y())
            if event.buttons() & Qt.MouseButton.LeftButton:
                dlat = 0.003 * \
                    (event.pos().y() - self.prevmousepos[1]) / (self.zoom * self.ar)
                dlon = 0.003 * \
                    (self.prevmousepos[0] - event.pos().x()) / \
                    (self.zoom * self.flat_earth)
                self.prevmousepos = (event.pos().x(), event.pos().y())
                self.panzoomchanged = True
                return self.panzoom(pan=(dlat, dlon))

        elif event.type() == QEvent.Type.TouchBegin:
            # Accept touch start to enable reception of follow-on touch update and touch end events
            event.accept()

        # Update pan/zoom to simulation thread only when the pan/zoom gesture is finished
        elif (event.type() == QEvent.Type.MouseButtonRelease or
              event.type() == QEvent.Type.TouchEnd) and self.panzoomchanged:
            self.panzoomchanged = False
            bs.net.send_event(b'PANZOOM', dict(pan=(self.panlat, self.panlon),
                                               zoom=self.zoom, ar=self.ar, absolute=True))
            self.panzoom_event.emit(True)
        elif int(event.type()) == 216:
            # 216 is screen change event, but doesn't exist (yet) in pyqt as enum
            self.pxratio = self.devicePixelRatio()
            self.shaderset.set_pixel_ratio(self.pxratio)
            return super().event(event)
        else:
            return super().event(event)
        
        # If we get here, the event was a mouse/trackpad event. Emit it to interested children
        self.mouse_event.emit(event)

        # For all other events call base class event handling
        return True

