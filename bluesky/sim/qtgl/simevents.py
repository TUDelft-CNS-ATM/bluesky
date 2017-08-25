""" Definition of data content to be transferred between GUI and Sim tasks,
    these defintions are used on both sides of the communication """
try:
    from PyQt5.QtCore import QEvent
except ImportError:
    from PyQt4.QtCore import QEvent

NUMEVENTS = 16
SetNodeIdType, SetActiveNodeType, AddNodeType, SimStateEventType, BatchEventType, \
    PanZoomEventType, ACDataEventType, SimInfoEventType, StackTextEventType, \
    StackInitEventType, ShowDialogEventType, DisplayFlagEventType, \
    RouteDataEventType, DisplayShapeEventType, \
    SimQuitEventType, AMANEventType = list(range(1000, 1000 + NUMEVENTS))

class EventBase(QEvent):
    def __init__(self, eventtype):
        super(EventBase, self).__init__(eventtype)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)


class SimStateEvent(EventBase):
    init, op, hold, end = list(range(4))

    def __init__(self, state):
        super(SimStateEvent, self).__init__(SimStateEventType)
        self.state = state


class BatchEvent(EventBase):
    def __init__(self, scentime, scencmd):
        super(BatchEvent, self).__init__(BatchEventType)
        self.scentime = scentime
        self.scencmd  = scencmd


class DisplayFlagEvent(EventBase):
    def __init__(self, switch='', argument=''):
        super(DisplayFlagEvent, self).__init__(DisplayFlagEventType)
        self.switch = switch
        self.argument = argument


class SimInfoEvent(EventBase):
    def __init__(self, sys_freq, simdt, simt, simtclock, n_ac, mode, scenname):
        super(SimInfoEvent, self).__init__(SimInfoEventType)
        self.sys_freq  = sys_freq
        self.simdt     = simdt
        self.simt      = simt
        self.simtclock = simtclock
        self.n_ac      = n_ac
        self.mode      = mode
        self.scenname  = scenname


class StackTextEvent(EventBase):
    def __init__(self, disptext='', cmdtext='', sender_id=None):
        super(StackTextEvent, self).__init__(StackTextEventType)
        self.disptext = disptext
        self.cmdtext = cmdtext
        self.sender_id = sender_id


class StackInitEvent(EventBase):
    def __init__(self, stackdict):
        super(StackInitEvent, self).__init__(StackInitEventType)
        self.stackdict = stackdict


class ShowDialogEvent(EventBase):
    # Types of dialog
    filedialog_type = 0
    docwin_type = 1

    def __init__(self, dialog_type=0, **extraattr):
        super(ShowDialogEvent, self).__init__(ShowDialogEventType)
        self.dialog_type = dialog_type
        self.__dict__.update(extraattr)


class RouteDataEvent(EventBase):
    aclat  = []
    wplat  = []
    wplon  = []
    wpalt  = []
    wpspd  = []
    wpname = []
    iactwp = -1
    acid  = ""

    def __init__(self):
        super(RouteDataEvent, self).__init__(RouteDataEventType)


class DisplayShapeEvent(EventBase):
    name = ''
    data = None

    def __init__(self, name, data=None):
        super(DisplayShapeEvent, self).__init__(DisplayShapeEventType)
        self.name = name
        self.data = data


class ACDataEvent(EventBase):
    lat = lon = alt = tas = trk = vs = iconf = confcpalat = confcpalon = id = []
    nconf_tot = nlos_tot  = nconf_exp = nlos_exp  = nconf_cur = nlos_cur = 0

    def __init__(self):
        super(ACDataEvent, self).__init__(ACDataEventType)


class AMANEvent(EventBase):
    ids = iafs = eats = etas = delays = rwys = spdratios = []

    def __init__(self):
        super(AMANEvent, self).__init__(AMANEventType)


class PanZoomEvent(EventBase):
    def __init__(self, pan=None, zoom=None, origin=None, absolute=False):
        super(PanZoomEvent, self).__init__(PanZoomEventType)
        self.pan      = pan
        self.origin   = origin
        self.zoom     = zoom
        self.absolute = absolute


class SimQuitEvent(EventBase):
    def __init__(self):
        super(SimQuitEvent, self).__init__(SimQuitEventType)
