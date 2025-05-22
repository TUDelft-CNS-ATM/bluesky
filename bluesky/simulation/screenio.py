""" ScreenIO is a screen proxy on the simulation side for the QTGL implementation of BlueSky."""
import time
import numpy as np

# Local imports
import bluesky as bs
from bluesky import stack
from bluesky.core import Entity
from bluesky.tools import aero
from bluesky.core.walltime import Timer
from bluesky.network.publisher import state_publisher, StatePublisher


# =========================================================================
# Settings
# =========================================================================
# Update rate of simulation info messages [Hz]
SIMINFO_RATE = 1

# Update rate of aircraft update messages [Hz]
ACUPDATE_RATE = 5


class ScreenIO(Entity):
    """Class within sim task which sends/receives data to/from GUI task"""

    pub_defwpt = StatePublisher('DEFWPT', collect=True)
    pub_route = StatePublisher('ROUTEDATA')

    # =========================================================================
    # Functions
    # =========================================================================
    def __init__(self):
        super().__init__()
        # Screen state defaults
        self.def_pan     = (0.0, 0.0)
        self.def_zoom    = 1.0
        self.route_all   = ""

        # Screen state overrides per client
        self.client_route = dict()

        # Dicts of custom aircraft and group colors
        self.custacclr = dict()
        self.custgrclr = dict()

        # Timing bookkeeping counters
        self.prevtime    = 0.0
        self.samplecount = 0
        self.prevcount   = 0

        # Output event timers
        self.slow_timer = Timer(1000 // SIMINFO_RATE)
        self.slow_timer.timeout.connect(self.send_siminfo)
        self.slow_timer.timeout.connect(self.send_route_data)

    def update(self):
        if bs.sim.state == bs.OP:
            self.samplecount += 1

    def reset(self):
        self.client_route = dict()
        self.route_all = ''
        self.custacclr = dict()
        self.custgrclr = dict()
        self.samplecount = 0
        self.prevcount   = 0
        self.prevtime    = 0.0

        self.def_pan = (0.0, 0.0)
        self.def_zoom = 1.0

    def showroute(self, acid):
        ''' Toggle show route for this aircraft '''
        if not stack.sender():
            self.route_all = acid
            self.client_route.clear()
        else:
            self.client_route[stack.sender()] = acid
        return True

    def addnavwpt(self, name, lat, lon):
        ''' Add custom waypoint to visualization '''
        self.pub_defwpt.send_append(custwplbl=name, custwplat=lat, custwplon=lon)
        return True

    def removenavwpt(self, name):
        ''' Remove custom waypoint to visualization '''
        self.pub_defwpt.send_delete(custwplbl=name)
        return True

    def show_file_dialog(self):
        bs.net.send(b'SHOWDIALOG', dict(dialog='OPENFILE'))
        return ''

    def show_cmd_doc(self, cmd=''):
        bs.net.send(b'SHOWDIALOG', dict(dialog='DOC', args=cmd))

    @stack.commandgroup(annotations='txt,color', aliases=('COLOR', 'COL'))
    def colour(self, name, r, g, b):
        ''' Set custom color for aircraft or shape. '''
        if name in bs.traf.groups:
            groupmask = bs.traf.groups.groups[name]
            self.custgrclr[groupmask] = (r, g, b)
        elif name in bs.traf.id:
            self.custacclr[name] = (r, g, b)

    # =========================================================================
    # Slots
    # =========================================================================
    def send_siminfo(self):
        t  = time.time()
        dt = np.maximum(t - self.prevtime, 0.00001)  # avoid divide by 0
        speed = (self.samplecount - self.prevcount) / dt * bs.sim.simdt
        self.prevtime  = t
        self.prevcount = self.samplecount
        bs.net.send(b'SIMINFO', (speed, bs.sim.simdt, bs.sim.simt,
            str(bs.sim.utc.replace(microsecond=0)), bs.traf.ntraf, bs.sim.state, stack.get_scenname()))
        

    @state_publisher(topic='TRAILS', dt=1000 // SIMINFO_RATE, send_type='extend')
    def send_trails(self):
        # Trails, send only new line segments to be added
        if bs.traf.trails.active and len(bs.traf.trails.newlat0) > 0:
            data = dict(traillat0=bs.traf.trails.newlat0,
                        traillon0=bs.traf.trails.newlon0,
                        traillat1=bs.traf.trails.newlat1,
                        traillon1=bs.traf.trails.newlon1)
            bs.traf.trails.clearnew()
            return data

    @state_publisher(topic='ACDATA', dt=1000 // ACUPDATE_RATE)
    def send_aircraft_data(self):
        data = dict()
        data['simt']       = bs.sim.simt
        data['id']         = bs.traf.id
        data['lat']        = bs.traf.lat
        data['lon']        = bs.traf.lon
        data['alt']        = bs.traf.alt
        data['tas']        = bs.traf.tas
        data['cas']        = bs.traf.cas
        data['gs']         = bs.traf.gs
        data['ingroup']    = bs.traf.groups.ingroup
        data['inconf'] = bs.traf.cd.inconf
        data['tcpamax'] = bs.traf.cd.tcpamax
        data['rpz'] = bs.traf.cd.rpz
        data['nconf_cur'] = len(bs.traf.cd.confpairs_unique)
        data['nconf_tot'] = len(bs.traf.cd.confpairs_all)
        data['nlos_cur'] = len(bs.traf.cd.lospairs_unique)
        data['nlos_tot'] = len(bs.traf.cd.lospairs_all)
        data['trk']        = bs.traf.trk
        data['vs']         = bs.traf.vs
        data['vmin']       = bs.traf.perf.vmin
        data['vmax']       = bs.traf.perf.vmax

        # Transition level as defined in traf
        data['translvl']   = bs.traf.translvl

        # Send casmachthr for route visualization
        data['casmachthr']    = aero.casmach_thr

        # ASAS resolutions for visualization. Only send when evaluated
        data['asastas']  = bs.traf.cr.tas
        data['asastrk']  = bs.traf.cr.trk

        # Aircraft (group) color
        if self.custacclr:
            data['custacclr'] = self.custacclr
        if self.custgrclr:
            data['custgrclr'] = self.custgrclr

        return data

    def send_route_data(self):
        ''' Send route data to client(s) '''
        # Case 1: A route is selected by one or more specific clients
        if self.client_route:
            for sender, acid in self.client_route.items():
                self._sendrte(sender, acid)
            # Check if there are other senders and also a scenario-selected route
            if self.route_all:
                remclients = bs.sim.clients.difference(self.client_route.keys())
                for sender in remclients:
                    self._sendrte(sender, self.route_all)
        # Case 2: only a route selected from scenario file:
        # Broadcast the same route to everyone
        elif self.route_all:
            self._sendrte(b'*', self.route_all)
        
    def _sendrte(self, sender, acid):
        ''' Local shorthand function to send route. '''
        data               = dict()
        data['acid']       = acid
        idx   = bs.traf.id2idx(acid)
        if idx >= 0:
            route          = bs.traf.ap.route[idx]
            data['iactwp'] = route.iactwp

            # We also need the corresponding aircraft position
            data['aclat']  = bs.traf.lat[idx]
            data['aclon']  = bs.traf.lon[idx]

            data['wplat']  = route.wplat
            data['wplon']  = route.wplon

            data['wpalt']  = route.wpalt
            data['wpspd']  = route.wpspd

            data['wpname'] = route.wpname

        self.pub_route.send_replace((sender or b'C'), **data)