''' BlueSky OpenGL line and polygon (areafilter) drawing. '''
import numpy as np
from bluesky.core import Signal
from bluesky.network import context as ctx
from bluesky.network.subscriber import subscriber
from bluesky.stack import command
from bluesky.ui import palette
from bluesky.ui.polytools import PolygonSet
from bluesky.ui.qtgl import console
from bluesky.ui.qtgl import glhelpers as glh
from bluesky.network.sharedstate import ActData
from bluesky.network.common import ActionType


palette.set_default_colours(
    polys=(0, 0, 255),
    previewpoly=(0, 204, 255)
) 

# Static defines
POLYPREV_SIZE = 500 # Max number of vertices in preview off new edited polygon
POLY_SIZE = 20000   # Max total number of vertices when summing all polygon vertices


class Poly(glh.RenderObject, layer=-20):
    ''' Poly OpenGL object. '''

    # Per remote node attributes
    show_poly: ActData[int] = ActData(1)
    polys: ActData[dict] = ActData(group='poly')
    bufdata: ActData[dict] = ActData()

    @command
    def showpoly(self, flag:int=None):
        ''' Toggle drawing of polygon shapes between off, outline, and outline+fill. '''
        # Cycle aircraft label through detail level 0,1,2
        if flag is None:
            self.show_poly = (self.show_poly + 1) % 3

        # Or use the argument if it is an integer
        else:
            self.show_poly = min(2,max(0,flag))

    def __init__(self, parent=None):
        super().__init__(parent)
        # Polygon preview object
        self.polyprev = glh.VertexArrayObject(glh.gl.GL_LINE_LOOP)

        # Fixed polygons
        self.allpolys = glh.VertexArrayObject(glh.gl.GL_LINES)
        self.allpfill = glh.VertexArrayObject(glh.gl.GL_TRIANGLES)

        self.prevmousepos = (0, 0)

        Signal('cmdline_stacked').connect(self.cmdline_stacked)
        Signal('radarmouse').connect(self.previewpoly)

    def create(self):
        # self.polyprev.create(vertex=POLYPREV_SIZE * 8,
        #                      color=palette.previewpoly, usage=glh.gl.GL_DYNAMIC_DRAW)
        self.polyprev.create(vertex=POLYPREV_SIZE * 8,
                             color=palette.previewpoly, usage=glh.GLBuffer.UsagePattern.DynamicDraw)
        self.allpolys.create(vertex=POLY_SIZE * 16, color=POLY_SIZE * 8)
        self.allpfill.create(vertex=POLY_SIZE * 24,
                             color=np.append(palette.polys, 50))

    def draw(self):
        # Send the (possibly) updated global uniforms to the buffer
        self.shaderset.set_vertex_scale_type(self.shaderset.VERTEX_IS_LATLON)

        # --- DRAW THE MAP AND COASTLINES ---------------------------------------------
        # Map and coastlines: don't wrap around in the shader
        self.shaderset.enable_wrap(False)

        # --- DRAW PREVIEW SHAPE (WHEN AVAILABLE) -----------------------------
        self.polyprev.draw()

        # --- DRAW CUSTOM SHAPES (WHEN AVAILABLE) -----------------------------
        if self.show_poly > 0:
            self.allpolys.draw()
            if self.show_poly > 1:
                self.allpfill.draw()
        
    def cmdline_stacked(self, cmd, args):
        if cmd in ['AREA', 'BOX', 'POLY', 'POLYGON', 'CIRCLE', 'LINE', 'POLYLINE']:
            self.polyprev.set_vertex_count(0)

    # def previewpoly(self, shape_type, data_in=None):
    def previewpoly(self, mouseevent):
        if mouseevent.type() != mouseevent.Type.MouseMove:
            return
        mousepos = (mouseevent.pos().x(), mouseevent.pos().y())
        # Check if we are updating a preview poly
        if mousepos != self.prevmousepos:
            cmd = console.get_cmd()
            nargs = len(console.get_args())
            if cmd in ['AREA', 'BOX', 'POLY', 'POLYLINE',
                        'POLYALT', 'POLYGON', 'CIRCLE', 'LINE'] and nargs >= 2:
                self.prevmousepos = mousepos
                try:
                    # get the largest even number of points
                    start = 0 if cmd == 'AREA' else 3 if cmd == 'POLYALT' else 1
                    end = ((nargs - start) // 2) * 2 + start
                    data = [float(v) for v in console.get_args()[start:end]]
                    data += self.glsurface.pixelCoordsToLatLon(*mousepos)
                    self.glsurface.makeCurrent()
                    if cmd is None:
                        self.polyprev.set_vertex_count(0)
                        return
                    if cmd in ['BOX', 'AREA']:
                        # For a box (an area is a box) we need to add two additional corners
                        polydata = np.zeros(8, dtype=np.float32)
                        polydata[0:2] = data[0:2]
                        polydata[2:4] = data[2], data[1]
                        polydata[4:6] = data[2:4]
                        polydata[6:8] = data[0], data[3]
                    else:
                        polydata = np.array(data, dtype=np.float32)

                    if cmd[-4:] == 'LINE':
                        self.polyprev.set_primitive_type(glh.gl.GL_LINE_STRIP)
                    else:
                        self.polyprev.set_primitive_type(glh.gl.GL_LINE_LOOP)

                    self.polyprev.update(vertex=polydata)


                except ValueError:
                    pass

    @subscriber(topic='POLY')
    def update_poly_data(self, data):
        if ctx.action in (ActionType.Reset, ActionType.ActChange):
            # Simulation reset: Clear all entries
            self.bufdata.clear()
            self.allpolys.set_vertex_count(0)
            self.allpfill.set_vertex_count(0)
            names = data.polys.keys()
        
        # The data argument passed to this subscriber contains all poly
        # data. We only need the current updates, which are in ctx.action_content
        elif ctx.action == ActionType.Delete:
            # Delete action contains a list of poly names to delete
            names = ctx.action_content['polys']
        else:
            # All other updates contain a dict with names as keys
            # and updated/new polys as items
            names = ctx.action_content['polys'].keys()

        # We're either updating a polygon, or deleting it.
        for name in names:
            # Always delete the old processed data
            self.bufdata.pop(name, None)

            if ctx.action != ctx.action.Delete:
                polydata = data.polys[name]
                try:
                    shape = polydata['shape']
                    coordinates = polydata['coordinates']
                    color = polydata.get('color', palette.polys)
                    self.bufdata[name] = self.genbuffers(shape, coordinates, color)
                except:
                    print("Could not process incoming poly data")
                
        if self.bufdata:
            self.glsurface.makeCurrent()
            contours, fills, colors = zip(*self.bufdata.values())
            # Create contour buffer with color
            self.allpolys.update(vertex=np.concatenate(contours),
                                    color=np.concatenate(colors))

            # Create fill buffer
            self.allpfill.update(vertex=np.concatenate(fills))
        else:
            self.allpolys.set_vertex_count(0)
            self.allpfill.set_vertex_count(0)

    @staticmethod
    def genbuffers(shape, coordinates, color=None):
        ''' Generate outline, fill, and colour buffers for given shape. '''
        # Break up polyline list of (lat,lon)s into separate line segments
        if shape == 'LINE' or shape[:4] == 'POLY':
            # Input data is list or array: [lat0,lon0,lat1,lon1,lat2,lon2,lat3,lon3,..]
            newdata = np.array(coordinates, dtype=np.float32)

        elif shape == 'BOX':
            # Convert box coordinates into polyline list
            # BOX: 0 = lat0, 1 = lon0, 2 = lat1, 3 = lon1 , use bounding box
            newdata = np.array([coordinates[0], coordinates[1],
                            coordinates[0], coordinates[3],
                            coordinates[2], coordinates[3],
                            coordinates[2], coordinates[1]], dtype=np.float32)

        elif shape == 'CIRCLE':
            # Input data is latctr,lonctr,radius[nm]
            # Convert circle into polyline list

            # Circle parameters
            Rearth = 6371000.0 # radius of the Earth [m]
            numPoints = 72 # number of straight line segments that make up the circrle

            # Inputs
            lat0 = coordinates[0] # latitude of the center of the circle [deg]
            lon0 = coordinates[1] # longitude of the center of the circle [deg]
            Rcircle = coordinates[2] * 1852.0 # radius of circle [NM]

            # Compute flat Earth correction at the center of the experiment circle
            coslatinv = 1.0 / np.cos(np.deg2rad(lat0))

            # compute the x and y coordinates of the circle
            angles    = np.linspace(0.0, 2.0 * np.pi, numPoints)   # ,endpoint=True) # [rad]

            # Calculate the circle coordinates in lat/lon degrees.
            # Use flat-earth approximation to convert from cartesian to lat/lon.
            latCircle = lat0 + np.rad2deg(Rcircle * np.sin(angles) / Rearth)  # [deg]
            lonCircle = lon0 + np.rad2deg(Rcircle * np.cos(angles) * coslatinv / Rearth)  # [deg]

            # make the data array in the format needed to plot circle
            newdata = np.empty(2 * numPoints, dtype=np.float32)  # Create empty array
            newdata[0::2] = latCircle  # Fill array lat0,lon0,lat1,lon1....
            newdata[1::2] = lonCircle

        # Create polygon contour buffer
        # Distinguish between an open and a closed contour.
        # If this is a closed contour, add the first vertex again at the end
        # and add a fill shape
        if shape[-4:] == 'LINE':
            contourbuf = np.empty(2 * len(newdata) - 4, dtype=np.float32)
            contourbuf[0::4]   = newdata[0:-2:2]  # lat
            contourbuf[1::4]   = newdata[1:-2:2]  # lon
            contourbuf[2::4] = newdata[2::2]  # lat
            contourbuf[3::4] = newdata[3::2]  # lon
            fillbuf = np.array([], dtype=np.float32)
        else:
            contourbuf = np.empty(2 * len(newdata), dtype=np.float32)
            contourbuf[0::4]   = newdata[0::2]  # lat
            contourbuf[1::4]   = newdata[1::2]  # lon
            contourbuf[2:-2:4] = newdata[2::2]  # lat
            contourbuf[3:-3:4] = newdata[3::2]  # lon
            contourbuf[-2:]    = newdata[0:2]
            pset = PolygonSet()
            pset.addContour(newdata)
            fillbuf = np.array(pset.vbuf, dtype=np.float32)

        # Define color buffer for outline
        defclr = tuple(color or palette.polys) + (255,)
        colorbuf = np.array(len(contourbuf) // 2 * defclr, dtype=np.uint8)

        # Store new or updated polygon by name, and concatenated with the
        # other polys
        return contourbuf, fillbuf, colorbuf
