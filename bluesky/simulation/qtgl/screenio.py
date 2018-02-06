""" ScreenIO is a screen proxy on the simulation side for the QTGL implementation of BlueSky."""
import time
import numpy as np

# Local imports
import bluesky as bs
from bluesky import stack
from bluesky.tools import Timer


class ScreenIO(object):
    """Class within sim task which sends/receives data to/from GUI task"""

    # =========================================================================
    # Settings
    # =========================================================================
    # Update rate of simulation info messages [Hz]
    siminfo_rate = 1

    # Update rate of aircraft update messages [Hz]
    acupdate_rate = 5

    # =========================================================================
    # Functions
    # =========================================================================
    def __init__(self):
        # Keep track of the important parameters of the screen state
        # (We receive these through events from the gui)
        self.ctrlat      = 0.0
        self.ctrlon      = 0.0
        self.scrzoom     = 1.0
        self.scrar       = 1.0

        self.route_acid  = None

        # Timing bookkeeping counters
        self.prevtime    = 0.0
        self.samplecount = 0
        self.prevcount   = 0

        # Output event timers
        self.slow_timer = Timer()
        self.slow_timer.timeout.connect(self.send_siminfo)
        self.slow_timer.timeout.connect(self.send_route_data)
        self.slow_timer.start(int(1000 / self.siminfo_rate))

        self.fast_timer = Timer()
        self.fast_timer.timeout.connect(self.send_aircraft_data)
        self.fast_timer.start(int(1000 / self.acupdate_rate))

    def update(self):
        if bs.sim.state == bs.OP:
            self.samplecount += 1

    def reset(self):
        self.samplecount = 0
        self.prevcount   = 0
        self.prevtime    = 0.0

        # Communicate reset to gui
        bs.sim.send_event(b'RESET', b'ALL')


    def echo(self, text):
        bs.sim.send_event(b'ECHO', text)

    def cmdline(self, text):
        bs.sim.send_event(b'CMDLINE', text)

    def getviewlatlon(self):
        lat0 = self.ctrlat - 1.0 / (self.scrzoom * self.scrar)
        lat1 = self.ctrlat + 1.0 / (self.scrzoom * self.scrar)
        lon0 = self.ctrlon - 1.0 / (self.scrzoom * np.cos(np.radians(self.ctrlat)))
        lon1 = self.ctrlon + 1.0 / (self.scrzoom * np.cos(np.radians(self.ctrlat)))
        return lat0, lat1, lon0, lon1

    def zoom(self, zoom, absolute=True):
        if absolute:
            self.scrzoom = zoom
        else:
            self.scrzoom *= zoom
        bs.sim.send_event(b'PANZOOM', dict(zoom=zoom, absolute=absolute))

    def pan(self, *args):
        ''' Move center of display, relative of to absolute position lat,lon '''
        if args[0] == "LEFT":
            self.ctrlon -= 0.5
        elif args[0] == "RIGHT":
            self.ctrlon += 0.5
        elif args[0] == "UP" or args[0] == "ABOVE":
            self.ctrlat += 0.5
        elif args[0] == "DOWN":
            self.ctrlat -= 0.5
        else:
            self.ctrlat, self.ctrlon = args

        bs.sim.send_event(b'PANZOOM', dict(pan=(self.ctrlat, self.ctrlon), absolute=True))

    def shownd(self, acid):
        bs.sim.send_event(b'SHOWND', acid)

    def symbol(self):
        bs.sim.send_event(b'DISPLAYFLAG', dict(flag='SYM'))

    def feature(self, switch, argument=None):
        bs.sim.send_event(b'DISPLAYFLAG', dict(flag=switch, args=argument))

    def trails(self,sw):
        bs.sim.send_event(b'DISPLAYFLAG', dict(flag='TRAIL', args=sw))

    def showroute(self, acid):
        ''' Toggle show route for this aircraft '''
        self.route_acid = acid
        return True

    def addnavwpt(self, name, lat, lon):
        ''' Add custom waypoint to visualization '''
        bs.sim.send_event(b'DEFWPT', dict(name=name, lat=lat, lon=lon))
        return True

    def show_file_dialog(self):
        bs.sim.send_event(b'SHOWDIALOG', dict(dialog='OPENFILE'))
        return ''

    def show_cmd_doc(self, cmd=''):
        bs.sim.send_event(b'SHOWDIALOG', dict(dialog='DOC', args=cmd))

    def filteralt(self, *args):
        bs.sim.send_event(b'DISPLAYFLAG', dict(flag='FILTERALT', args=args))

    def objappend(self, objtype, objname, data):
        """Add a drawing object to the radar screen using the following inpouts:
           objtype: "LINE"/"POLY" /"BOX"/"CIRCLE" = string with type of object
           objname: string with a name as key for reference
           objdata: lat,lon data, depending on type:
                    POLY/LINE: lat0,lon0,lat1,lon1,lat2,lon2,....
                    BOX : lat0,lon0,lat1,lon1   (bounding box coordinates)
                    CIRCLE: latctr,lonctr,radiusnm  (circle parameters)
        """
        bs.sim.send_event(b'SHAPE', dict(name=objname, shape=objtype, data=data))

    def event(self, eventname, eventdata, sender_id):
        if eventname == b'PANZOOM':
            self.ctrlat  = eventdata['pan'][0]
            self.ctrlon  = eventdata['pan'][1]
            self.scrzoom = eventdata['zoom']
            self.scrar   = eventdata['ar']
            return True

        return False

    # =========================================================================
    # Slots
    # =========================================================================
    def send_siminfo(self):
        t  = time.time()
        dt = np.maximum(t - self.prevtime, 0.00001)  # avoid divide by 0
        speed = (self.samplecount - self.prevcount) / dt * bs.sim.simdt
        bs.sim.send_stream(b'SIMINFO', (speed, bs.sim.simdt, bs.sim.simt,
            bs.sim.simtclock, bs.traf.ntraf, bs.sim.state, stack.get_scenname()))
        self.prevtime  = t
        self.prevcount = self.samplecount

    def send_aircraft_data(self):
        data               = dict()
        data['simt']       = bs.sim.simt
        data['id']         = bs.traf.id
        data['lat']        = bs.traf.lat
        data['lon']        = bs.traf.lon
        data['alt']        = bs.traf.alt
        data['tas']        = bs.traf.tas
        data['cas']        = bs.traf.cas
        data['iconf']      = bs.traf.asas.iconf
        data['confcpalat'] = bs.traf.asas.latowncpa
        data['confcpalon'] = bs.traf.asas.lonowncpa
        data['trk']        = bs.traf.hdg
        data['vs']         = bs.traf.vs
        data['vmin']       = bs.traf.asas.vmin
        data['vmax']       = bs.traf.asas.vmax

        # Trails, send only new line segments to be added
        data['swtrails']  = bs.traf.trails.active
        data['traillat0'] = bs.traf.trails.newlat0
        data['traillon0'] = bs.traf.trails.newlon0
        data['traillat1'] = bs.traf.trails.newlat1
        data['traillon1'] = bs.traf.trails.newlon1
        bs.traf.trails.clearnew()

        # Last segment which is being built per aircraft
        data['traillastlat']   = bs.traf.trails.lastlat
        data['traillastlon']   = bs.traf.trails.lastlon

        # Conflict statistics
        data['nconf_tot']  = len(bs.traf.asas.conflist_all)
        data['nlos_tot']   = len(bs.traf.asas.LOSlist_all)
        data['nconf_exp']  = len(bs.traf.asas.conflist_exp)
        data['nlos_exp']   = len(bs.traf.asas.LOSlist_exp)
        data['nconf_cur']  = len(bs.traf.asas.conflist_now)
        data['nlos_cur']   = len(bs.traf.asas.LOSlist_now)

        # Transition level as defined in traf
        data['translvl']   = bs.traf.translvl

        # ASAS resolutions for visualization. Only send when evaluated
        if bs.traf.asas.asaseval:
            data['asasn']  = bs.traf.asas.asasn
            data['asase']  = bs.traf.asas.asase
        else:
            data['asasn']  = np.zeros(bs.traf.ntraf, dtype=np.float32)
            data['asase']  = np.zeros(bs.traf.ntraf, dtype=np.float32)

        bs.sim.send_stream(b'ACDATA', data)

    def send_route_data(self):
        if self.route_acid:
            data               = dict()
            data['acid']       = self.route_acid
            idx   = bs.traf.id2idx(self.route_acid)
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

            bs.sim.send_stream(b'ROUTEDATA', data)  # Send route data to GUI
