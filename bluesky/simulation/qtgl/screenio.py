""" ScreenIO is a screen proxy on the simulation side for the QTGL implementation of BlueSky."""
import time
import numpy as np

# Local imports
import bluesky as bs
from bluesky import stack
import bluesky.manager as manager
from bluesky.manager import Timer
from .simevents import ACDataEvent, RouteDataEvent, PanZoomEvent, \
                        SimInfoEvent, StackTextEvent, ShowDialogEvent, DisplayFlagEvent, \
                        PanZoomEventType, DisplayShapeEvent


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
        if bs.sim.state == bs.sim.op:
            self.samplecount += 1

    def reset(self):
        self.samplecount = 0
        self.prevcount   = 0
        self.prevtime    = 0.0

        # Communicate reset to gui
        manager.send_event(DisplayFlagEvent('RESET', 'ALL'))


    def echo(self, text):
        manager.send_event(StackTextEvent(disptext=text))

    def cmdline(self, text):
        manager.send_event(StackTextEvent(cmdtext=text))

    def getviewlatlon(self):
        lat0 = self.ctrlat - 1.0 / self.scrzoom
        lat1 = self.ctrlat + 1.0 / self.scrzoom
        lon0 = self.ctrlon - 1.0 / self.scrzoom
        lon1 = self.ctrlon + 1.0 / self.scrzoom
        return lat0, lat1, lon0, lon1

    def zoom(self, zoom, absolute=True):
        if absolute:
            self.scrzoom = zoom
        else:
            self.scrzoom *= zoom
        manager.send_event(PanZoomEvent(zoom=zoom, absolute=absolute))

    def symbol(self):
        manager.send_event(DisplayFlagEvent('SYM'))

    def trails(self,sw):
        manager.send_event(DisplayFlagEvent('TRAIL',sw))

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

        manager.send_event(PanZoomEvent(pan=(self.ctrlat, self.ctrlon), absolute=True))

    def showroute(self, acid):
        ''' Toggle show route for this aircraft '''
        self.route_acid = acid
        return True

    def addnavwpt(self, name, lat, lon):
        ''' Add custom waypoint to visualization '''
        manager.send_event(DisplayFlagEvent('DEFWPT', (name, lat, lon)))
        return True

    def showssd(self, *param):
        ''' Conflict prevention display
            Show solution space diagram, indicating potential conflicts'''
        manager.send_event(DisplayFlagEvent('SSD', param))

    def show_file_dialog(self):
        manager.send_event(ShowDialogEvent())
        return ''

    def show_cmd_doc(self, cmd=''):
        manager.send_event(ShowDialogEvent(1, cmd=cmd))

    def feature(self, switch, argument=''):
        manager.send_event(DisplayFlagEvent(switch, argument))

    def filteralt(self, *args):
        manager.send_event(DisplayFlagEvent('FILTERALT', args))

    def objappend(self, objtype, objname, data_in):
        """Add a drawing object to the radar screen using the following inpouts:
           objtype: "LINE"/"POLY" /"BOX"/"CIRCLE" = string with type of object
           objname: string with a name as key for reference
           objdata: lat,lon data, depending on type:
                    POLY/LINE: lat0,lon0,lat1,lon1,lat2,lon2,....
                    BOX : lat0,lon0,lat1,lon1   (bounding box coordinates)
                    CIRCLE: latctr,lonctr,radiusnm  (circle parameters)
        """
        if data_in is None:
            # This is an object delete event
            data = None

        elif objtype == 'LINE' or objtype[:4] == 'POLY':
            # Input data is laist or array: [lat0,lon0,lat1,lon1,lat2,lon2,lat3,lon3,..]
            data = np.array(data_in, dtype=np.float32)

        elif objtype == 'BOX':
            # Convert box coordinates into polyline list
            # BOX: 0 = lat0, 1 = lon0, 2 = lat1, 3 = lon1 , use bounding box
            data = np.array([data_in[0], data_in[1],
                             data_in[0], data_in[3],
                             data_in[2], data_in[3],
                             data_in[2], data_in[1]], dtype=np.float32)

        elif objtype == 'CIRCLE':
            # Input data is latctr,lonctr,radius[nm]
            # Convert circle into polyline list

            # Circle parameters
            Rearth = 6371000.0             # radius of the Earth [m]
            numPoints = 72                 # number of straight line segments that make up the circrle

            # Inputs
            lat0 = data_in[0]              # latitude of the center of the circle [deg]
            lon0 = data_in[1]              # longitude of the center of the circle [deg]
            Rcircle = data_in[2] * 1852.0  # radius of circle [NM]

            # Compute flat Earth correction at the center of the experiment circle
            coslatinv = 1.0 / np.cos(np.deg2rad(lat0))

            # compute the x and y coordinates of the circle
            angles    = np.linspace(0.0, 2.0 * np.pi, numPoints)   # ,endpoint=True) # [rad]

            # Calculate the circle coordinates in lat/lon degrees.
            # Use flat-earth approximation to convert from cartesian to lat/lon.
            latCircle = lat0 + np.rad2deg(Rcircle * np.sin(angles) / Rearth)  # [deg]
            lonCircle = lon0 + np.rad2deg(Rcircle * np.cos(angles) * coslatinv / Rearth)  # [deg]

            # make the data array in the format needed to plot circle
            data = np.empty(2 * numPoints, dtype=np.float32)  # Create empty array
            data[0::2] = latCircle  # Fill array lat0,lon0,lat1,lon1....
            data[1::2] = lonCircle

        manager.send_event(DisplayShapeEvent(objname, data))

    def event(self, event, sender_id):
        if event.type() == PanZoomEventType:
            self.ctrlat  = event.pan[0]
            self.ctrlon  = event.pan[1]
            self.scrzoom = event.zoom
            return True

        return False

    # =========================================================================
    # Slots
    # =========================================================================
    def send_siminfo(self):
        t  = time.time()
        dt = np.maximum(t - self.prevtime, 0.00001)  # avoid divide by 0
        speed = (self.samplecount - self.prevcount) / dt * bs.sim.simdt
        manager.send_event(SimInfoEvent(speed, bs.sim.simdt, bs.sim.simt,
            bs.sim.simtclock, bs.traf.ntraf, bs.sim.state, stack.get_scenname()))
        self.prevtime  = t
        self.prevcount = self.samplecount

    def send_aircraft_data(self):
        data            = ACDataEvent()
        data.simt       = bs.sim.simt
        data.id         = bs.traf.id
        data.lat        = bs.traf.lat
        data.lon        = bs.traf.lon
        data.alt        = bs.traf.alt
        data.tas        = bs.traf.tas
        data.cas        = bs.traf.cas
        data.iconf      = bs.traf.asas.iconf
        data.confcpalat = bs.traf.asas.latowncpa
        data.confcpalon = bs.traf.asas.lonowncpa
        data.trk        = bs.traf.hdg
        data.vs         = bs.traf.vs

        # Trails, send only new line segments to be added
        data.swtrails  = bs.traf.trails.active
        data.traillat0 = bs.traf.trails.newlat0
        data.traillon0 = bs.traf.trails.newlon0
        data.traillat1 = bs.traf.trails.newlat1
        data.traillon1 = bs.traf.trails.newlon1
        bs.traf.trails.clearnew()

        # Last segment which is being built per aircraft
        data.traillastlat   = bs.traf.trails.lastlat
        data.traillastlon   = bs.traf.trails.lastlon

        # Conflict statistics
        data.nconf_tot  = len(bs.traf.asas.conflist_all)
        data.nlos_tot   = len(bs.traf.asas.LOSlist_all)
        data.nconf_exp  = len(bs.traf.asas.conflist_exp)
        data.nlos_exp   = len(bs.traf.asas.LOSlist_exp)
        data.nconf_cur  = len(bs.traf.asas.conflist_now)
        data.nlos_cur   = len(bs.traf.asas.LOSlist_now)

        # Transition level as defined in traf
        data.translvl   = bs.traf.translvl

        manager.sendEvent(data)

    def send_route_data(self):
        if self.route_acid:
            data               = RouteDataEvent()
            data.acid          = self.route_acid
            idx   = bs.traf.id2idx(self.route_acid)
            if idx >= 0:
                route          = bs.traf.ap.route[idx]
                data.iactwp    = route.iactwp

                # We also need the corresponding aircraft position
                data.aclat     = bs.traf.lat[idx]
                data.aclon     = bs.traf.lon[idx]

                data.wplat     = route.wplat
                data.wplon     = route.wplon

                data.wpalt     = route.wpalt
                data.wpspd     = route.wpspd

                data.wpname    = route.wpname

            manager.send_event(data)  # Send route data to GUI
