""" Definition of custom QEvent objects for QtGL gui. """
from PyQt5.QtCore import QEvent

NUMCUSTOMEVENTS    = 2
ACDataEventType    = 1000
RouteDataEventType = 1001


class RouteDataEvent(QEvent):
    def __init__(self, data=None):
        super(RouteDataEvent, self).__init__(RouteDataEventType)
        self.aclat  = []
        self.wplat  = []
        self.wplon  = []
        self.wpalt  = []
        self.wpspd  = []
        self.wpname = []
        self.iactwp = -1
        self.acid   = ''

        # Update values
        if data:
            self.__dict__.update(data)


class ACDataEvent(QEvent):
    def __init__(self, data=None):
        super(ACDataEvent, self).__init__(ACDataEventType)
        self.lat        = []
        self.lon        = []
        self.alt        = []
        self.tas        = []
        self.trk        = []
        self.vs         = []
        self.iconf      = []
        self.confcpalat = []
        self.confcpalon = []
        self.id         = []
        self.nconf_tot  = 0
        self.nlos_tot   = 0
        self.nconf_exp  = 0
        self.nlos_exp   = 0
        self.nconf_cur  = 0
        self.nlos_cur   = 0
        self.translvl   = 0.0

        # Update values
        if data:
            self.__dict__.update(data)
