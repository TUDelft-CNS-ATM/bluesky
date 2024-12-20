''' BlueSky navdata OpenGL visualisation object. '''
import numpy as np
import bluesky as bs
from bluesky.core import Signal
from bluesky.network import context as ctx
from bluesky.network.subscriber import subscriber
from bluesky.network.sharedstate import ActData
from bluesky.stack import command
from bluesky.ui.qtgl import glhelpers as glh
from bluesky import settings
from bluesky.ui.loadvisuals import load_aptsurface
from bluesky.ui import palette


# Register settings defaults
settings.set_variable_defaults(
    gfx_path='graphics',
    text_size=13, apt_size=10,
    wpt_size=10)

palette.set_default_colours(
    aptlabel=(220, 250, 255),
    aptsymbol=(148, 178, 235),
    pavement=(160, 160, 160),
    runways=(100, 100, 100),
    taxiways=(100, 100, 100),
    thresholds=(255, 255, 255),
    wptlabel=(220, 250, 255),
    wptsymbol=(148, 178, 235)
)

# Static defines
CUSTWP_SIZE = 1000


class Navdata(glh.RenderObject, layer=-10):
    ''' Navdata OpenGL object. '''

    # Per remote node attributes
    show_wpt: ActData[int] = ActData(1)
    show_apt: ActData[int] = ActData(1)
    pan: ActData[list[float]] = ActData([0.0, 0.0], group='panzoom')
    zoom: ActData[float] = ActData(1.0, group='panzoom')

    @command
    def showwpt(self, flag:int=None):
        ''' Toggle drawing of waypoints. '''
        # TODO: add to SWRAD
        # flag == 'VOR' or flag == 'WPT' or flag == 'WP' or flag == 'NAV':
        # Cycle waypoint visualisation through detail level 0,1,2
        if flag is None:
            self.show_wpt = (self.show_wpt + 1) % 3

        # Or use the argument if it is an integer
        else:
            self.show_wpt = min(2,max(0,flag))

    @command
    def showapt(self, flag:int=None):
        ''' Toggle drawing of waypoints. '''
        # Cycle waypoint visualisation through detail level 0,1,2
        # Airport: 0 = None, 1 = Large, 2= All
        if flag is None:
            self.show_apt = (self.show_apt + 1) % 3

        # Or use the argument if it is an integer
        else:
            self.show_apt = min(2,max(0,flag))

    def __init__(self, parent=None):
        super().__init__(parent)
        self.custwplblbuf = glh.GLBuffer()

        # ------- Airport graphics -----------------------
        self.runways = glh.VertexArrayObject(glh.gl.GL_TRIANGLES)
        self.thresholds = glh.VertexArrayObject(glh.gl.GL_TRIANGLES)
        self.taxiways = glh.VertexArrayObject(glh.gl.GL_TRIANGLES)
        self.pavement = glh.VertexArrayObject(glh.gl.GL_TRIANGLES)

        self.apt_ctrlat = []
        self.apt_ctrlon = []
        self.apt_indices = []
        self.apt_inrange = np.array([])

        self.nwaypoints = 0
        self.nnavaids = 0
        self.ncustwpts = 0
        self.waypoints = glh.VertexArrayObject(glh.gl.GL_LINE_LOOP)
        self.wptlabels = glh.Text(settings.text_size, (5, 1))
        self.customwp = glh.VertexArrayObject(glh.gl.GL_LINE_LOOP)
        self.customwplbl = glh.Text(settings.text_size, (10, 1))

        self.nairports = []
        self.airports = glh.VertexArrayObject(glh.gl.GL_LINE_LOOP)
        self.aptlabels = glh.Text(settings.text_size, (4, 1))

        self.vbuf_asphalt, self.vbuf_concrete, self.vbuf_runways, self.vbuf_rwythr, \
            self.apt_ctrlat, self.apt_ctrlon, self.apt_indices = load_aptsurface()

        Signal('state-changed.panzoom').connect(self.panzoom)

    def panzoom(self, data, finished=True):
        if ctx.action is None or ctx.action == ctx.action.Reset:
            return
        # TODO: need good way to set defaults!!
        if not hasattr(data, 'zoom'):
            data.zoom = 1.0
        if not hasattr(data, 'pan'):
            data.pan = [0.0, 0.0]
        if data.zoom >= 1.0:
            # Airports may be visible when zoom > 1: in this case, update the list of indicates
            # of airports that need to be drawn
            ll_range = max(1.5 / data.zoom, 1.0)
            indices = np.logical_and(np.abs(self.apt_ctrlat - data.pan[0])
                                    <= ll_range, np.abs(self.apt_ctrlon - data.pan[1]) <= ll_range)
            self.apt_inrange = self.apt_indices[indices]
        else:
            self.apt_inrange = np.array([])
    
    @subscriber
    def defwpt(self, data):
        ''' Receive custom waypoint data and add to visualisation. '''
        if not data.valid():
            self.ncustwpts = 0
            return

        self.ncustwpts = len(data.custwplbl)
        self.customwp.update(lat=np.array(data.custwplat, dtype=np.float32),
                                lon=np.array(data.custwplon, dtype=np.float32))
        lbl = [n[:10].ljust(10) for n in data.custwplbl]
        self.custwplblbuf.update(
            np.array(lbl, dtype=np.bytes_))

    def create(self):
        apt_size = settings.apt_size
        wpt_size = settings.wpt_size
        # self.custwplblbuf.create(CUSTWP_SIZE * 10, usage=glh.gl.GL_STATIC_DRAW)
        self.custwplblbuf.create(CUSTWP_SIZE * 10, usage=glh.GLBuffer.UsagePattern.StaticDraw)

        # Load vertex data
        
        self.runways.create(vertex=self.vbuf_runways, color=palette.runways)
        self.thresholds.create(vertex=self.vbuf_rwythr, color=palette.thresholds)
        self.taxiways.create(vertex=self.vbuf_asphalt, color=palette.taxiways)
        self.pavement.create(vertex=self.vbuf_concrete, color=palette.pavement)
        del self.vbuf_asphalt, self.vbuf_concrete, self.vbuf_runways, self.vbuf_rwythr

        # ------- Waypoints ------------------------------
        wptvertices = np.array([(0.0, 0.5 * wpt_size), (-0.5 * wpt_size, -0.5 * wpt_size),
                                (0.5 * wpt_size, -0.5 * wpt_size)], dtype=np.float32)  # a triangle
        self.nwaypoints = len(bs.navdb.wplat)
        self.waypoints.create(vertex=wptvertices, color=palette.wptsymbol,
                              n_instances=self.nwaypoints)
        # Sort based on id string length
        llid = sorted(zip(bs.navdb.wpid, bs.navdb.wplat,
                          bs.navdb.wplon), key=lambda i: len(i[0]) > 3)
        wpidlst, wplat, wplon = zip(*llid)
        self.waypoints.set_attribs(lat=np.array(wplat, dtype=np.float32), lon=np.array(
            wplon, dtype=np.float32), instance_divisor=1)
        wptids = ''
        self.nnavaids = 0
        for wptid in wpidlst:
            if len(wptid) <= 3:
                self.nnavaids += 1
            wptids += wptid[:5].ljust(5)
        npwpids = np.array(wptids, dtype=np.bytes_)
        self.wptlabels.create(npwpids, self.waypoints.lat,
                              self.waypoints.lon, palette.wptlabel,
                              (wpt_size, 0.5 * wpt_size), instanced=True)

        self.customwp.create(vertex=self.waypoints.vertex, vertex_count=3, color=palette.wptsymbol)
        self.customwp.set_attribs(
            lat=CUSTWP_SIZE * 4, lon=CUSTWP_SIZE * 4, instance_divisor=1)
        self.customwplbl.create(self.custwplblbuf,
                                self.customwp.lat, self.customwp.lon,
                                palette.wptlabel, (wpt_size, 0.5 * wpt_size),
                                instanced=True)

        # ------- Airports -------------------------------
        aptvertices = np.array([(-0.5 * apt_size, -0.5 * apt_size),
                                (0.5 * apt_size, -0.5 * apt_size),
                                (0.5 * apt_size, 0.5 * apt_size),
                                (-0.5 * apt_size, 0.5 * apt_size)], dtype=np.float32)  # a square
        self.nairports = len(bs.navdb.aptlat)
        self.airports.create(vertex=aptvertices, color=palette.aptsymbol,
                             n_instances=self.nairports)
        indices = bs.navdb.aptype.argsort()
        aplat = np.array(bs.navdb.aptlat[indices], dtype=np.float32)
        aplon = np.array(bs.navdb.aptlon[indices], dtype=np.float32)
        aptypes = bs.navdb.aptype[indices]
        apnames = np.array(bs.navdb.aptid)
        apnames = apnames[indices]
        # The number of large, large+med, and large+med+small airports
        self.nairports = [aptypes.searchsorted(
            2), aptypes.searchsorted(3), self.nairports]

        self.airports.set_attribs(lat=aplat, lon=aplon, instance_divisor=1)
        aptids = ''
        for aptid in apnames:
            aptids += aptid.ljust(4)
        self.aptlabels.create(np.array(aptids, dtype=np.bytes_),
                              self.airports.lat, self.airports.lon,
                              palette.aptlabel, (apt_size, 0.5 * apt_size),
                              instanced=True)

    def draw(self):
        # Send the (possibly) updated global uniforms to the buffer
        self.shaderset.set_vertex_scale_type(self.shaderset.VERTEX_IS_LATLON)

        self.shaderset.enable_wrap(False)

        # --- DRAW AIRPORT DETAILS (RUNWAYS, TAXIWAYS, PAVEMENTS) -------------
        self.runways.draw()
        self.thresholds.draw()

        if self.zoom >= 1.0:
            for idx in self.apt_inrange:
                self.taxiways.draw(first_vertex=idx[0], vertex_count=idx[1])
                self.pavement.draw(first_vertex=idx[2], vertex_count=idx[3])

        self.shaderset.enable_wrap(True)
        self.shaderset.set_vertex_scale_type(self.shaderset.VERTEX_IS_SCREEN)

        if self.zoom >= 0.5 and self.show_apt == 1 or self.show_apt == 2:
            nairports = self.nairports[2]
        elif self.zoom >= 0.25 and self.show_apt == 1 or self.show_apt == 3:
            nairports = self.nairports[1]
        else:
            nairports = self.nairports[0]

        if self.zoom >= 3 and self.show_wpt == 1 or self.show_wpt == 2:
            nwaypoints = self.nwaypoints
        else:
            nwaypoints = self.nnavaids

        # Draw waypoint symbols
        if self.show_wpt:
            self.waypoints.draw(n_instances=nwaypoints)
            if self.ncustwpts > 0:
                self.customwp.draw(n_instances=self.ncustwpts)

        # Draw airport symbols
        if self.show_apt:
            self.airports.draw(n_instances=nairports)

        # Draw wpt/apt labels
        if self.show_apt:
            self.aptlabels.draw(n_instances=nairports)
        if self.show_wpt:
            self.wptlabels.draw(n_instances=nwaypoints)
            if self.ncustwpts > 0:
                self.customwplbl.draw(n_instances=self.ncustwpts)
