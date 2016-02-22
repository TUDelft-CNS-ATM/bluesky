try:
    # Try Qt5 first
    from PyQt5.QtCore import QObject, QTimer, pyqtSlot
    from PyQt5.QtCore import QCoreApplication as qapp
except ImportError:
    # Else fall back to Qt4
    from PyQt4.QtCore import QObject, QTimer, pyqtSlot
    from PyQt4.QtCore import QCoreApplication as qapp

import numpy as np
import time

# Local imports
from ...ui.qtgl import ACDataEvent, RouteDataEvent, PanZoomEvent, SimInfoEvent, StackTextEvent, \
                       ShowDialogEvent, DisplayFlagEvent, StackTextEventType, PanZoomEventType, \
                       DisplayShapeEvent, SimQuitEvent, AMANEvent


class ScreenIO(QObject):
    # =========================================================================
    # Settings
    # =========================================================================
    # Update rate of simulation info messages [Hz]
    siminfo_rate = 2

    # Update rate of aircraft update messages [Hz]
    acupdate_rate = 5

    # =========================================================================
    # Slots
    # =========================================================================
    @pyqtSlot()
    def send_siminfo(self):
        t  = time.time()
        dt = t - self.prevtime
        qapp.postEvent(qapp.instance(), SimInfoEvent((self.sim.samplecount - self.prevcount) / dt, self.sim.simdt, self.sim.simt, self.sim.traf.ntraf, self.sim.mode))
        self.prevtime  = t
        self.prevcount = self.sim.samplecount

    @pyqtSlot()
    def send_aircraft_data(self):
        data            = ACDataEvent()
        data.id         = list(self.sim.traf.id)
        data.lat        = np.array(self.sim.traf.lat, dtype=np.float32, copy=True)
        data.lon        = np.array(self.sim.traf.lon, dtype=np.float32, copy=True)
        data.alt        = np.array(self.sim.traf.alt, dtype=np.float32, copy=True)
        data.tas        = np.array(self.sim.traf.tas, dtype=np.float32, copy=True)
        data.iconf      = np.array(self.sim.traf.iconf, copy=True)
        data.confcpalat = np.array(self.sim.traf.dbconf.latowncpa, copy=True)
        data.confcpalon = np.array(self.sim.traf.dbconf.lonowncpa, copy=True)
        data.trk        = np.array(self.sim.traf.trk, dtype=np.float32, copy=True)
        qapp.postEvent(qapp.instance(), data)

    @pyqtSlot()
    def send_aman_data(self):
        # data            = AMANEvent()
        # data.ids        = self.sim.traf.AMAN.
        # data.iafs       = self.sim.traf.AMAN.
        # data.eats       = self.sim.traf.AMAN.
        # data.etas       = self.sim.traf.AMAN.
        # data.delays     = self.sim.traf.AMAN.
        # data.rwys       = self.sim.traf.AMAN.
        # data.spdratios  = self.sim.traf.AMAN. 
        # qapp.postEvent(qapp.instance(), data)
        pass

    # =========================================================================
    # Functions
    # =========================================================================
    def __init__(self, sim):
        super(ScreenIO, self).__init__()

        # Keep track of the important parameters of the screen state
        # (We receive these through events from the gui)
        self.ctrlat  = 0.0
        self.ctrlon  = 0.0
        self.scrzoom = 1.0

        # Keep reference to parent simulation object for access to simulation data
        self.sim = sim

        # Timing bookkeeping counters
        self.prevtime = 0.0
        self.prevcount = 0

        # Output event timers
        self.siminfo_timer = QTimer()
        self.siminfo_timer.timeout.connect(self.send_siminfo)
        self.siminfo_timer.timeout.connect(self.send_aman_data)
        self.siminfo_timer.start(1000/self.siminfo_rate)

        self.acupdate_timer = QTimer()
        self.acupdate_timer.timeout.connect(self.send_aircraft_data)
        self.acupdate_timer.start(1000/self.acupdate_rate)

    def moveToThread(self, target_thread):
        self.siminfo_timer.moveToThread(target_thread)
        self.acupdate_timer.moveToThread(target_thread)
        super(ScreenIO, self).moveToThread(target_thread)

    def echo(self, text):
        qapp.postEvent(qapp.instance(), StackTextEvent(text))

    def getviewlatlon(self):
        lat0 = self.ctrlat - 1.0 / self.scrzoom
        lat1 = self.ctrlat + 1.0 / self.scrzoom
        lon0 = self.ctrlon - 1.0 / self.scrzoom
        lon1 = self.ctrlon + 1.0 / self.scrzoom
        return lat0, lat1, lon0, lon1

    def zoom(self, zoomfac, absolute=False):
        if absolute:
            self.scrzoom = zoomfac
        else:
            self.scrzoom *= zoomfac
        qapp.postEvent(qapp.instance(), PanZoomEvent(zoom=zoomfac, absolute=absolute))

    def symbol(self):
        qapp.postEvent(qapp.instance(), DisplayFlagEvent('SYM'))
        return

    def pan(self, pan, absolute=False):
        if absolute:
            self.ctrlat = pan[0]
            self.ctrlon = pan[1]
        else:
            self.ctrlat += pan[0]
            self.ctrlon += pan[1]
        qapp.postEvent(qapp.instance(), PanZoomEvent(pan=pan, absolute=absolute))

    def showroute(self, acid):
        data       = RouteDataEvent()
        data.acidx = self.sim.traf.id2idx(acid)
        if data.acidx >= 0:
            route = self.sim.traf.route[data.acidx]
            n_segments     = len(route.wplat) + 1
            data.lat       = np.empty(n_segments, dtype=np.float32)
            data.lat[0]    = self.sim.traf.lat[data.acidx]
            data.lat[1:]   = route.wplat
            data.lon       = np.empty(n_segments, dtype=np.float32)
            data.lon[0]    = self.sim.traf.lon[data.acidx]
            data.lon[1:]   = route.wplon
            wptlabels      = []
            for i in range(1, n_segments):
                wptlabels += '%-8s' % route.wpname[i-1]
            data.wptlabels = np.array(wptlabels)
        qapp.postEvent(qapp.instance(), data)

    def showssd(self, param):
        if param == 'ALL' or param == 'OFF':
            qapp.postEvent(qapp.instance(), DisplayFlagEvent('SSD', param))
        else:
            idx = self.sim.traf.id2idx(param)
            if idx >= 0:
                qapp.postEvent(qapp.instance(), DisplayFlagEvent('SSD', idx))

    def show_file_dialog(self):
        qapp.postEvent(qapp.instance(), ShowDialogEvent())
        return ''

    def postQuit(self):
        qapp.postEvent(qapp.instance(), SimQuitEvent())
        return ''

    def feature(self, switch, argument=''):
        qapp.postEvent(qapp.instance(), DisplayFlagEvent(switch, argument))

    def objappend(self, objtype, objname, data_in):
        if data_in is None:
            # This is an object delete event
            data = None
        elif objtype == 1 or objtype == 4:
            # LINE(1) or POLY(4)
            data = np.array(data_in, dtype=np.float32)
        elif objtype == 2:
            # BOX
            data = np.array([data_in[0], data_in[1],
                             data_in[0], data_in[3],
                             data_in[2], data_in[3],
                             data_in[2], data_in[1]], dtype=np.float32)

        elif objtype == 3:
            # CIRCLE
            pass

        qapp.postEvent(qapp.instance(), DisplayShapeEvent(objname, data))

    def event(self, event):
        if event.type() == StackTextEventType:
            self.sim.stack.stack(event.text)

        elif event.type() == PanZoomEventType:
            self.ctrlat  = event.pan[0]
            self.ctrlon  = event.pan[1]
            self.scrzoom = event.zoom

        return True
