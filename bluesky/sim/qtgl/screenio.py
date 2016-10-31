try:
    # Try Qt5 first
    from PyQt5.QtCore import QObject, pyqtSlot
except ImportError:
    # Else fall back to Qt4
    from PyQt4.QtCore import QObject, pyqtSlot

import numpy as np
import time

# Local imports
from ... import stack
from timer import Timer
from simevents import ACDataEvent, RouteDataEvent, PanZoomEvent, \
                        SimInfoEvent, StackTextEvent, ShowDialogEvent, DisplayFlagEvent, \
                        PanZoomEventType, DisplayShapeEvent


class ScreenIO(QObject):
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
    def __init__(self, sim, manager):
        super(ScreenIO, self).__init__()

        # Keep track of the important parameters of the screen state
        # (We receive these through events from the gui)
        self.ctrlat      = 0.0
        self.ctrlon      = 0.0
        self.scrzoom     = 1.0

        self.route_acid  = None

        # Keep reference to parent simulation object for access to simulation data
        self.sim         = sim
        self.manager     = manager

        # Timing bookkeeping counters
        self.prevtime    = 0.0
        self.samplecount = 0
        self.prevcount   = 0

        # Output event timers
        self.slow_timer = Timer()
        self.slow_timer.timeout.connect(self.send_siminfo)
        self.slow_timer.timeout.connect(self.send_aman_data)
        self.slow_timer.timeout.connect(self.send_route_data)
        self.slow_timer.start(1000 / self.siminfo_rate)

        self.fast_timer = Timer()
        self.fast_timer.timeout.connect(self.send_aircraft_data)
        self.fast_timer.start(1000 / self.acupdate_rate)

    def update(self):
        if self.sim.state == self.sim.op:
            self.samplecount += 1

    def reset(self):
        self.samplecount = 0
        self.prevcount   = 0
        self.prevtime    = 0.0

    def echo(self, text):
        if self.manager.isActive():
            self.manager.sendEvent(StackTextEvent(disptext=text))

    def cmdline(self, text):
        if self.manager.isActive():
            self.manager.sendEvent(StackTextEvent(cmdtext=text))

    def getviewlatlon(self):
        lat0 = self.ctrlat - 1.0 / self.scrzoom
        lat1 = self.ctrlat + 1.0 / self.scrzoom
        lon0 = self.ctrlon - 1.0 / self.scrzoom
        lon1 = self.ctrlon + 1.0 / self.scrzoom
        return lat0, lat1, lon0, lon1

    def zoom(self, zoom, absolute=True):
        if self.manager.isActive():
            if absolute:
                self.scrzoom = zoom
            else:
                self.scrzoom *= zoom
            self.manager.sendEvent(PanZoomEvent(zoom=zoom, absolute=absolute))

    def symbol(self):
        if self.manager.isActive():
            self.manager.sendEvent(DisplayFlagEvent('SYM'))

    def pan(self, *args):
        if self.manager.isActive():
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

            self.manager.sendEvent(PanZoomEvent(pan=(self.ctrlat, self.ctrlon), absolute=True))

    def showroute(self, acid):
        self.route_acid = acid
        return True

    def showacinfo(self, acid, infotext):
        self.echo(infotext)
        self.showroute(acid)
        return True

    def showssd(self, param):
        if self.manager.isActive():
            if param == 'ALL' or param == 'OFF':
                self.manager.sendEvent(DisplayFlagEvent('SSD', param))
            else:
                idx = self.sim.traf.id2idx(param)
                if idx >= 0:
                    self.manager.sendEvent(DisplayFlagEvent('SSD', idx))

    def show_file_dialog(self):
        if self.manager.isActive():
            self.manager.sendEvent(ShowDialogEvent())
        return ''

    def feature(self, switch, argument=''):
        if self.manager.isActive():
            self.manager.sendEvent(DisplayFlagEvent(switch, argument))

    def objappend(self, objtype, objname, data_in):
        if data_in is None:
            # This is an object delete event
            data = None
        elif objtype == 'LINE' or objtype[:4] == 'POLY':
            data = np.array(data_in, dtype=np.float32)
        elif objtype == 'BOX':
            # BOX: 0 = lat0, 1 = lon0, 2 = lat1, 3 = lat1 , use bounding box  
            data = np.array([data_in[0], data_in[1],
                             data_in[0], data_in[3],
                             data_in[2], data_in[3],
                             data_in[2], data_in[1]], dtype=np.float32)

        elif objtype == 'CIRCLE':
            # parameters
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

        self.manager.sendEvent(DisplayShapeEvent(objname, data))

    def event(self, event):
        if event.type() == PanZoomEventType:
            self.ctrlat  = event.pan[0]
            self.ctrlon  = event.pan[1]
            self.scrzoom = event.zoom
            return True

        return False

    # =========================================================================
    # Slots
    # =========================================================================
    @pyqtSlot()
    def send_siminfo(self):
        t  = time.time()
        dt = np.maximum(t - self.prevtime, 0.00001)  # avoid divide by 0
        speed = (self.samplecount - self.prevcount) / dt * self.sim.simdt
        self.manager.sendEvent(SimInfoEvent(speed, self.sim.simdt, self.sim.simt,
            self.sim.simtclock, self.sim.traf.ntraf, self.sim.state, stack.get_scenname()))
        self.prevtime  = t
        self.prevcount = self.samplecount

    @pyqtSlot()
    def send_aircraft_data(self):
        if self.manager.isActive():
            data            = ACDataEvent()
            data.id         = self.sim.traf.id
            data.lat        = self.sim.traf.lat
            data.lon        = self.sim.traf.lon
            data.alt        = self.sim.traf.alt
            data.tas        = self.sim.traf.tas
            data.cas        = self.sim.traf.cas
            data.iconf      = self.sim.traf.asas.iconf
            data.confcpalat = self.sim.traf.asas.latowncpa
            data.confcpalon = self.sim.traf.asas.lonowncpa
            data.trk        = self.sim.traf.hdg

            # Conflict statistics
            data.nconf_tot  = len(self.sim.traf.asas.conflist_all)
            data.nlos_tot   = len(self.sim.traf.asas.LOSlist_all)
            data.nconf_exp  = len(self.sim.traf.asas.conflist_exp)
            data.nlos_exp   = len(self.sim.traf.asas.LOSlist_exp)
            data.nconf_cur  = len(self.sim.traf.asas.conflist_now)
            data.nlos_cur   = len(self.sim.traf.asas.LOSlist_now)

            self.manager.sendEvent(data)

    @pyqtSlot()
    def send_route_data(self):
        if self.manager.isActive() and self.route_acid is not None:
            data               = RouteDataEvent()
            data.acid          = self.route_acid
            idx   = self.sim.traf.id2idx(self.route_acid)
            if idx >= 0:
                route          = self.sim.traf.ap.route[idx]
                data.iactwp    = route.iactwp

                # We also need the corresponding aircraft position
                data.aclat     = self.sim.traf.lat[idx]
                data.aclon     = self.sim.traf.lon[idx]

                data.lat       = route.wplat
                data.lon       = route.wplon

                data.wptlabels = route.wpname

            self.manager.sendEvent(data)  # Send route data to GUI
            # Empty route acid string means no longer send route data
            if len(self.route_acid) == 0:
                self.route_acid = None

    @pyqtSlot()
    def send_aman_data(self):
        if self.manager.isActive():
            # data            = AMANEvent()
            # data.ids        = self.sim.traf.AMAN.
            # data.iafs       = self.sim.traf.AMAN.
            # data.eats       = self.sim.traf.AMAN.
            # data.etas       = self.sim.traf.AMAN.
            # data.delays     = self.sim.traf.AMAN.
            # data.rwys       = self.sim.traf.AMAN.
            # data.spdratios  = self.sim.traf.AMAN. 
            # self.manager.sendEvent(data)
            pass
