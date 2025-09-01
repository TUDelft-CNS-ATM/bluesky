''' Traffic OpenGL visualisation. '''
import numpy as np
from bluesky.ui.qtgl import glhelpers as glh

from bluesky.stack import command
from bluesky.tools import geo
from bluesky import settings
from bluesky.ui import palette
from bluesky.tools.aero import ft, nm, kts
from bluesky.network import context as ctx
from bluesky.network.subscriber import subscriber
from bluesky.network.sharedstate import ActData


# Register settings defaults
settings.set_variable_defaults(
    text_size=13, ac_size=16,
    asas_vmin=200.0, asas_vmax=500.0)

palette.set_default_colours(
    aircraft=(0, 255, 0),
    conflict=(255, 160, 0),
    route=(255, 0, 255),
    trails=(0, 255, 255))

# Static defines
MAX_NAIRCRAFT = 10000
MAX_NCONFLICTS = 25000
MAX_ROUTE_LENGTH = 500
ROUTE_SIZE = 500
TRAILS_SIZE = 1000000


class Traffic(glh.RenderObject, layer=100):
    ''' Traffic OpenGL object. '''
    # Per remote node attributes
    show_pz: ActData[bool] = ActData(False)
    show_traf: ActData[bool] = ActData(True)
    show_lbl: ActData[int] = ActData(2)
    ssd_all: ActData[bool] = ActData(False)
    ssd_conflicts: ActData[bool] = ActData(False)
    ssd_ownship: ActData[set] = ActData(set())
    altrange: ActData[tuple] = ActData(tuple())
    naircraft: ActData[int] = ActData(0)
    zoom: ActData[float] = ActData(1.0, group='panzoom')

    @command
    def showpz(self, flag:bool=None):
        ''' Toggle drawing of aircraft protected zones. '''
        # TODO: add to SWRAD (flag=SYM)
        self.show_pz = not self.show_pz if flag is None else flag

    @command
    def showtraf(self, flag:bool=None):
        ''' Toggle drawing of aircraft. '''
        self.show_traf = not self.show_traf if flag is None else flag

    @command
    def label(self, flag:int=None):
        ''' Toggle drawing of aircraft label. '''
        # Cycle aircraft label through detail level 0,1,2
        if flag is None:
            self.show_lbl = (self.show_lbl + 1) % 3

        # Or use the argument if it is an integer
        else:
            self.show_lbl = min(2,max(0,flag))

    @command(name='SSD')
    def showssd(self, *arg:'txt'):
        ''' Show/hide SSD (state-space diagram) for one or more, or all aircraft.
        
            The SSD shows the vector space of conflicting vs. conflict-free aircraft
            velocities.

            Arguments:
            ALL: Show SSD for all aircraft
            CONFLICTS: Show SSD only for aircraft in conflict
            OFF: Hide SSDs
            acid(s): Show/hide SSD for given aircraft
        '''
        if 'ALL' in arg:
            self.ssd_all      = True
            self.ssd_conflicts = False
        elif 'CONFLICTS' in arg:
            self.ssd_all      = False
            self.ssd_conflicts = True
        elif 'OFF' in arg:
            self.ssd_all      = False
            self.ssd_conflicts = False
            self.ssd_ownship = set()
        else:
            remove = self.ssd_ownship.intersection(arg)
            self.ssd_ownship = self.ssd_ownship.union(arg) - remove

    @command
    def filteralt(self, flag:bool=None, bottom:'alt'=-1e99, top:'alt'=1e99):
        ''' Display aircraft on only a selected range of altitudes

            Arguments:
            - flag: Turn altitude filtering ON/OFF
            - bottom: The lowest altitude in the visible range (optional)
            - top: The highest altitude in the visible range (optional)
        '''
        if flag is None:
            if not self.altrange:
                return True, f'The current altitude range is unlimited'
            elif self.altrange[0] < -1e90:
                return True, f'The current altitude range is limited below {self.altrange[1]} meters'
            elif self.altrange[1] > 1e90:
                return True, f'The current altitude range is limited above {self.altrange[0]} meters'
            else:
                return True, f'The current altitude range is limited between {self.altrange[0]} and {self.altrange[1]} meters'
        if flag:
            self.altrange = (bottom, top)        

    def __init__(self, parent=None):
        super().__init__(parent)
        self.initialized = False
        self.route_acid = ''
        self.asas_vmin = settings.asas_vmin
        self.asas_vmax = settings.asas_vmax
        self.hdg = glh.GLBuffer()
        self.rpz = glh.GLBuffer()
        self.lat = glh.GLBuffer()
        self.lon = glh.GLBuffer()
        self.alt = glh.GLBuffer()
        self.tas = glh.GLBuffer()
        self.color = glh.GLBuffer()
        self.lbl = glh.GLBuffer()
        self.asasn = glh.GLBuffer()
        self.asase = glh.GLBuffer()

        self.ssd = glh.VertexArrayObject(glh.gl.GL_POINTS, shader_type='ssd')
        self.protectedzone = glh.Circle()
        self.ac_symbol = glh.VertexArrayObject(glh.gl.GL_TRIANGLE_FAN)
        self.aclabels = glh.Text(settings.text_size, (8, 3))
        self.cpalines = glh.VertexArrayObject(glh.gl.GL_LINES)
        self.route = glh.VertexArrayObject(glh.gl.GL_LINES)
        self.routelbl = glh.Text(settings.text_size, (12, 2))
        self.rwaypoints = glh.VertexArrayObject(glh.gl.GL_LINE_LOOP)
        self.traillines = glh.VertexArrayObject(glh.gl.GL_LINES)

    def create(self):
        ac_size = settings.ac_size
        wpt_size = settings.wpt_size
        self.hdg.create(MAX_NAIRCRAFT * 4, glh.GLBuffer.UsagePattern.StreamDraw)
        self.lat.create(MAX_NAIRCRAFT * 4, glh.GLBuffer.UsagePattern.StreamDraw)
        self.lon.create(MAX_NAIRCRAFT * 4, glh.GLBuffer.UsagePattern.StreamDraw)
        self.alt.create(MAX_NAIRCRAFT * 4, glh.GLBuffer.UsagePattern.StreamDraw)
        self.tas.create(MAX_NAIRCRAFT * 4, glh.GLBuffer.UsagePattern.StreamDraw)
        self.color.create(MAX_NAIRCRAFT * 4, glh.GLBuffer.UsagePattern.StreamDraw)
        self.lbl.create(MAX_NAIRCRAFT * 24, glh.GLBuffer.UsagePattern.StreamDraw)
        self.asasn.create(MAX_NAIRCRAFT * 24, glh.GLBuffer.UsagePattern.StreamDraw)
        self.asase.create(MAX_NAIRCRAFT * 24, glh.GLBuffer.UsagePattern.StreamDraw)
        self.rpz.create(MAX_NAIRCRAFT * 4, glh.GLBuffer.UsagePattern.StreamDraw)

        self.ssd.create(lat1=self.lat, lon1=self.lon, alt1=self.alt,
                        tas1=self.tas, trk1=self.hdg)
        self.ssd.set_attribs(selssd=MAX_NAIRCRAFT, instance_divisor=1,
                             datatype=glh.gl.GL_UNSIGNED_BYTE, normalize=True)
        self.ssd.set_attribs(lat0=self.lat, lon0=self.lon,
                             alt0=self.alt, tas0=self.tas,
                             trk0=self.hdg, asasn=self.asasn,
                             asase=self.asase, instance_divisor=1)

        #self.protectedzone.create(radius=1.0)
        self.protectedzone.create(radius=0.5)
        self.protectedzone.set_attribs(lat=self.lat, lon=self.lon, scale=self.rpz,
                                       color=self.color, instance_divisor=1)

        acvertices = np.array([(0.0, 0.5 * ac_size), (-0.5 * ac_size, -0.5 * ac_size),
                               (0.0, -0.25 * ac_size), (0.5 * ac_size, -0.5 * ac_size)],
                              dtype=np.float32)
        self.ac_symbol.create(vertex=acvertices)

        self.ac_symbol.set_attribs(lat=self.lat, lon=self.lon, color=self.color,
                                   orientation=self.hdg, instance_divisor=1)

        self.aclabels.create(self.lbl, self.lat, self.lon, self.color,
                             (ac_size, -0.5 * ac_size), instanced=True)

        self.cpalines.create(vertex=MAX_NCONFLICTS * 16, color=palette.conflict, usage=glh.GLBuffer.UsagePattern.StreamDraw)

        # ------- Aircraft Route -------------------------
        self.route.create(vertex=ROUTE_SIZE * 8, color=palette.route, usage=glh.GLBuffer.UsagePattern.DynamicDraw)

        self.routelbl.create(ROUTE_SIZE * 24, ROUTE_SIZE * 4, ROUTE_SIZE * 4,
                             palette.route, (wpt_size, 0.5 * wpt_size), instanced=True)
        rwptvertices = np.array([(-0.2 * wpt_size, -0.2 * wpt_size),
                                 (0.0,            -0.8 * wpt_size),
                                 (0.2 * wpt_size, -0.2 * wpt_size),
                                 (0.8 * wpt_size,  0.0),
                                 (0.2 * wpt_size,  0.2 * wpt_size),
                                 (0.0,             0.8 * wpt_size),
                                 (-0.2 * wpt_size,  0.2 * wpt_size),
                                 (-0.8 * wpt_size,  0.0)], dtype=np.float32)
        self.rwaypoints.create(vertex=rwptvertices, color=palette.route)
        self.rwaypoints.set_attribs(lat=self.routelbl.lat, lon=self.routelbl.lon, instance_divisor=1)

        # # --------Aircraft Trails------------------------------------------------
        self.traillines.create(vertex=TRAILS_SIZE * 16, color=palette.trails)
        self.initialized = True

    def draw(self):
        ''' Draw all traffic graphics. '''
        # Get data for active node
        if self.naircraft == 0 or not self.show_traf:
            return

        # Send the (possibly) updated global uniforms to the buffer
        self.shaderset.set_vertex_scale_type(self.shaderset.VERTEX_IS_LATLON)
        self.shaderset.enable_wrap(False)

        self.route.draw()
        self.cpalines.draw()
        self.traillines.draw()

        # --- DRAW THE INSTANCED AIRCRAFT SHAPES ------------------------------
        # update wrap longitude and direction for the instanced objects
        self.shaderset.enable_wrap(True)

        # PZ circles only when they are bigger than the A/C symbols
        if self.show_pz and self.zoom >= 0.15:
            self.shaderset.set_vertex_scale_type(
                self.shaderset.VERTEX_IS_METERS)
            self.protectedzone.draw(n_instances=self.naircraft)

        self.shaderset.set_vertex_scale_type(self.shaderset.VERTEX_IS_SCREEN)

        # Draw traffic symbols
        self.ac_symbol.draw(n_instances=self.naircraft)

        if self.routelbl.n_instances:
            self.rwaypoints.draw(n_instances=self.routelbl.n_instances)
            self.routelbl.draw()

        if self.show_lbl:
            self.aclabels.draw(n_instances=self.naircraft)

        # SSD
        if self.ssd_all or self.ssd_conflicts or len(self.ssd_ownship) > 0:
            ssd_shader = glh.ShaderSet.get_shader('ssd')
            ssd_shader.bind()
            glh.gl.glUniform3f(ssd_shader.uniforms['Vlimits'].loc, self.asas_vmin **
                           2, self.asas_vmax ** 2, self.asas_vmax)
            glh.gl.glUniform1i(ssd_shader.uniforms['n_ac'].loc, self.naircraft)
            self.ssd.draw(vertex_count=self.naircraft,
                          n_instances=self.naircraft)

    @subscriber(topic='TRAILS')
    def update_trails_data(self, data):
        ''' Update GPU buffers with route data from simulation. '''
        if not self.initialized:
            return
        if ctx.action == ctx.action.Reset or ctx.action == ctx.action.ActChange:# TODO hack
            # Simulation reset: Clear all entries
            self.traillines.set_vertex_count(0)
            return
        if len(data.traillat0) > 0:
            self.glsurface.makeCurrent()
            self.traillines.set_vertex_count(len(data.traillat0))
            if len(data.traillat0) > 0:
                self.traillines.update(vertex=np.array(
                        list(zip(data.traillat0, data.traillon0,
                                data.traillat1, data.traillon1)), dtype=np.float32))
        else:
            self.traillines.set_vertex_count(0)

    @subscriber(topic='ROUTEDATA', actonly=True)
    def update_route_data(self, data):
        ''' Update GPU buffers with route data from simulation. '''
        if not self.initialized:
            return
        if ctx.action in [ctx.action.Reset, ctx.action.ActChange, ctx.action.Delete]:
            # Simulation reset: Clear all entries
            self.route.set_vertex_count(0)
            self.routelbl.n_instances = 0
            return
        self.glsurface.makeCurrent()
        self.route_acid = data.acid
        if data.acid != "" and len(data.wplat) > 0:
            nsegments = len(data.wplat)
            data.iactwp = min(max(0, data.iactwp), nsegments - 1)
            self.routelbl.n_instances = nsegments
            self.route.set_vertex_count(2 * nsegments)
            routedata = np.empty(4 * nsegments, dtype=np.float32)
            routedata[0:4] = [data.aclat, data.aclon,
                              data.wplat[data.iactwp], data.wplon[data.iactwp]]

            routedata[4::4] = data.wplat[:-1]
            routedata[5::4] = data.wplon[:-1]
            routedata[6::4] = data.wplat[1:]
            routedata[7::4] = data.wplon[1:]

            self.route.update(vertex=routedata)
            wpname = ''
            for wp, alt, spd in zip(data.wpname, data.wpalt, data.wpspd):
                if alt < 0. and spd < 0.:
                    txt = wp[:12].ljust(24)  # No second line
                else:
                    txt = wp[:12].ljust(12)  # Two lines
                    if alt < 0:
                        txt += "-----/"
                    # TODO: get from sim
                    elif alt > 5000.0 * ft:
                        FL = int(round((alt / (100. * ft))))
                        txt += "FL%03d/" % FL
                    else:
                        txt += "%05d/" % int(round(alt / ft))

                    # Speed
                    if spd < 0:
                        txt += "--- "
                    # TODO: get from sim
                    elif spd > settings.casmach_threshold:
                        txt += "%03d" % int(round(spd / kts))
                    else:
                        txt += f"M{spd:.2f}" # Mach number

                wpname += txt.ljust(24)  # Fill out with spaces
            self.routelbl.update(texdepth=np.array(wpname.encode('ascii', 'ignore')),
                                 lat=np.array(data.wplat, dtype=np.float32),
                                 lon=np.array(data.wplon, dtype=np.float32))
        else:
            self.route.set_vertex_count(0)
            self.routelbl.n_instances = 0

    @subscriber(topic='ACDATA', actonly=True)
    def update_aircraft_data(self, data):
        ''' Update GPU buffers with new aircraft simulation data. '''
        if not self.initialized:
            return
        if ctx.action == ctx.action.Reset or ctx.action == ctx.action.ActChange:# TODO hack
            # Simulation reset: Clear all entries
            self.naircraft = 0
            return
        self.glsurface.makeCurrent()
        if self.altrange:
            idx = np.where(
                (data.alt >= self.altrange[0]) * (data.alt <= self.altrange[1]))
            data.lat = data.lat[idx]
            data.lon = data.lon[idx]
            data.trk = data.trk[idx]
            data.alt = data.alt[idx]
            data.tas = data.tas[idx]
            data.vs = data.vs[idx]
            data.rpz = data.rpz[idx]
        naircraft = len(data.lat)
        self.naircraft = naircraft
        # self.asas_vmin = data.vmin # TODO: array should be attribute not uniform
        # self.asas_vmax = data.vmax

        if naircraft == 0:
            self.cpalines.set_vertex_count(0)
        else:
            # Update data in GPU buffers
            self.lat.update(np.array(data.lat, dtype=np.float32))
            self.lon.update(np.array(data.lon, dtype=np.float32))
            self.hdg.update(np.array(data.trk, dtype=np.float32))
            self.alt.update(np.array(data.alt, dtype=np.float32))
            self.tas.update(np.array(data.tas, dtype=np.float32))
            self.rpz.update(np.array(data.rpz, dtype=np.float32))
            if hasattr(data, 'asasn') and hasattr(data, 'asase'):
                self.asasn.update(np.array(data.asasn, dtype=np.float32))
                self.asase.update(np.array(data.asase, dtype=np.float32))

            # CPA lines to indicate conflicts
            ncpalines = np.count_nonzero(data.inconf)

            cpalines = np.zeros(4 * ncpalines, dtype=np.float32)
            self.cpalines.set_vertex_count(2 * ncpalines)

            # Labels and colors
            rawlabel = ''
            color = np.empty(
                (min(naircraft, MAX_NAIRCRAFT), 4), dtype=np.uint8)
            selssd = np.zeros(naircraft, dtype=np.uint8)
            confidx = 0

            custacclr = getattr(data, 'custacclr', dict())
            custgrclr = getattr(data, 'custgrclr', dict())

            zdata = zip(data.id, data.ingroup, data.inconf, data.tcpamax, data.trk, data.gs,
                        data.cas, data.vs, data.alt, data.lat, data.lon)
            for i, (acid, ingroup, inconf, tcpa,
                    trk, gs, cas, vs, alt, lat, lon) in enumerate(zdata):
                if i >= MAX_NAIRCRAFT:
                    break

                # Make label: 3 lines of 8 characters per aircraft
                if self.show_lbl >= 1:
                    rawlabel += '%-8s' % acid[:8]
                    if self.show_lbl == 2:
                        if alt <= data.translvl:
                            rawlabel += '%-5d' % int(alt / ft + 0.5)
                        else:
                            rawlabel += 'FL%03d' % int(alt / ft / 100. + 0.5)
                        vsarrow = 30 if vs > 0.25 else 31 if vs < -0.25 else 32
                        rawlabel += '%1s  %-8d' % (chr(vsarrow),
                                                   int(cas / kts + 0.5))
                    else:
                        rawlabel += 16 * ' '

                if inconf:
                    if self.ssd_conflicts:
                        selssd[i] = 255
                    color[i, :] = palette.conflict + (255,)
                    lat1, lon1 = geo.qdrpos(lat, lon, trk, tcpa * gs / nm)
                    cpalines[4 * confidx: 4 * confidx +
                             4] = [lat, lon, lat1, lon1]
                    confidx += 1
                else:
                    # Get custom color if available, else default
                    rgb = palette.aircraft
                    if ingroup:
                        for groupmask, groupcolor in custgrclr.items():
                            if ingroup & groupmask:
                                rgb = groupcolor
                                break
                    rgb = custacclr.get(acid, rgb)
                    color[i, :] = tuple(rgb) + (255,)

                #  Check if aircraft is selected to show SSD
                if self.ssd_all or acid in self.ssd_ownship:
                    selssd[i] = 255

            if len(self.ssd_ownship) > 0 or self.ssd_conflicts or self.ssd_all:
                self.ssd.update(selssd=selssd)
            self.cpalines.update(vertex=cpalines)
            self.color.update(color)
            self.lbl.update(np.array(rawlabel.encode('utf8'), dtype=np.bytes_))
            
            # If there is a visible route, update the start position
            if self.route_acid in data.id:
                idx = data.id.index(self.route_acid)
                self.route.vertex.update(np.array([data.lat[idx], data.lon[idx]], dtype=np.float32))
