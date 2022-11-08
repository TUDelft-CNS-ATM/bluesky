''' BlueSky navdata OpenGL visualisation object. '''
import numpy as np
import bluesky as bs
from bluesky import Signal
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

        bs.net.actnodedata_changed.connect(self.actdata_changed)
        Signal('panzoom').connect(self.on_panzoom_signal)

    def on_panzoom_signal(self, finished):
        self.actdata_changed(0, bs.net.get_nodedata(), ('PANZOOM',))

    def actdata_changed(self, nodeid, nodedata, changed_elems):
        if 'PANZOOM' in changed_elems:
            # TODO panzoom state should always go via nodedata
            if nodedata.zoom >= 1.0:
                # Airports may be visible when zoom > 1: in this case, update the list of indicates
                # of airports that need to be drawn
                ll_range = max(1.5 / nodedata.zoom, 1.0)
                indices = np.logical_and(np.abs(self.apt_ctrlat - nodedata.pan[0])
                                     <= ll_range, np.abs(self.apt_ctrlon - nodedata.pan[1]) <= ll_range)
                self.apt_inrange = self.apt_indices[indices]
            else:
                self.apt_inrange = np.array([])
        if 'CUSTWPT' in changed_elems:
            if nodedata.custwplbl:
                self.customwp.update(lat=nodedata.custwplat,
                                     lon=nodedata.custwplon)
                self.custwplblbuf.update(
                    np.array(nodedata.custwplbl, dtype=np.string_))
            self.ncustwpts = len(nodedata.custwplat)

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
        npwpids = np.array(wptids, dtype=np.string_)
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
        self.aptlabels.create(np.array(aptids, dtype=np.string_),
                              self.airports.lat, self.airports.lon,
                              palette.aptlabel, (apt_size, 0.5 * apt_size),
                              instanced=True)

    def draw(self):
        actdata = bs.net.get_nodedata()
        # Send the (possibly) updated global uniforms to the buffer
        self.shaderset.set_vertex_scale_type(self.shaderset.VERTEX_IS_LATLON)

        self.shaderset.enable_wrap(False)

        # --- DRAW AIRPORT DETAILS (RUNWAYS, TAXIWAYS, PAVEMENTS) -------------
        self.runways.draw()
        self.thresholds.draw()

        if actdata.zoom >= 1.0:
            for idx in self.apt_inrange:
                self.taxiways.draw(first_vertex=idx[0], vertex_count=idx[1])
                self.pavement.draw(first_vertex=idx[2], vertex_count=idx[3])

        self.shaderset.enable_wrap(True)
        self.shaderset.set_vertex_scale_type(self.shaderset.VERTEX_IS_SCREEN)

        if actdata.zoom >= 0.5 and actdata.show_apt == 1 or actdata.show_apt == 2:
            nairports = self.nairports[2]
        elif actdata.zoom >= 0.25 and actdata.show_apt == 1 or actdata.show_apt == 3:
            nairports = self.nairports[1]
        else:
            nairports = self.nairports[0]

        if actdata.zoom >= 3 and actdata.show_wpt == 1 or actdata.show_wpt == 2:
            nwaypoints = self.nwaypoints
        else:
            nwaypoints = self.nnavaids

        # Draw waypoint symbols
        if actdata.show_wpt:
            self.waypoints.draw(n_instances=nwaypoints)
            if self.ncustwpts > 0:
                self.customwp.draw(n_instances=self.ncustwpts)

        # Draw airport symbols
        if actdata.show_apt:
            self.airports.draw(n_instances=nairports)

        # Draw wpt/apt labels
        if actdata.show_apt:
            self.aptlabels.draw(n_instances=nairports)
        if actdata.show_wpt:
            self.wptlabels.draw(n_instances=nwaypoints)
            if self.ncustwpts > 0:
                self.customwplbl.draw(n_instances=self.ncustwpts)
