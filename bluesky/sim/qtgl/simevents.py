try:
    from PyQt5.QtCore import QEvent
except ImportError:
    from PyQt4.QtCore import QEvent

NUMEVENTS = 15
SetNodeIdType, SetActiveNodeType, AddNodeType, SimStateEventType, BatchEventType, PanZoomEventType, ACDataEventType, \
    SimInfoEventType, StackTextEventType, ShowDialogEventType, \
    DisplayFlagEventType, RouteDataEventType, DisplayShapeEventType, \
    SimQuitEventType, AMANEventType = range(1000, 1000 + NUMEVENTS)

""" Definition of data content to be transferred between GUI and Sim tasks,
    these defintions are used on both sides of the communication """


class SimStateEvent(QEvent):
    init, op, hold, end = range(4)

    def __init__(self, state):
        super(SimStateEvent, self).__init__(SimStateEventType)
        self.state = state


class BatchEvent(QEvent):
    def __init__(self, scentime, scencmd):
        super(BatchEvent, self).__init__(BatchEventType)
        self.scentime = scentime
        self.scencmd  = scencmd


class DisplayFlagEvent(QEvent):
    def __init__(self, switch='', argument=''):
        super(DisplayFlagEvent, self).__init__(DisplayFlagEventType)
        self.switch = switch
        self.argument = argument


class SimInfoEvent(QEvent):
    def __init__(self, sys_freq, simdt, simt, simtclock, n_ac, mode, scenname):
        super(SimInfoEvent, self).__init__(SimInfoEventType)
        self.sys_freq  = sys_freq
        self.simdt     = simdt
        self.simt      = simt
        self.simtclock = simtclock
        self.n_ac      = n_ac
        self.mode      = mode
        self.scenname  = scenname


class StackTextEvent(QEvent):
    def __init__(self, disptext='', cmdtext=''):
        super(StackTextEvent, self).__init__(StackTextEventType)
        self.disptext = disptext
        self.cmdtext = cmdtext


class ShowDialogEvent(QEvent):
    # Types of dialog
    filedialog_type = 0

    def __init__(self, dialog_type=0):
        super(ShowDialogEvent, self).__init__(ShowDialogEventType)
        self.dialog_type = dialog_type


class RouteDataEvent(QEvent):
    aclat = aclon = wplat = wplon = wpname = []
    iactwp = -1
    acid = ""

    def __init__(self):
        super(RouteDataEvent, self).__init__(RouteDataEventType)


class DisplayShapeEvent(QEvent):
    name = ''
    data = None

    def __init__(self, name, data=None):
        super(DisplayShapeEvent, self).__init__(DisplayShapeEventType)
        self.name = name
        self.data = data


class ACDataEvent(QEvent):
    lat = lon = alt = tas = trk = iconf = confcpalat = confcpalon = id = []
    nconf_tot = nlos_tot  = nconf_exp = nlos_exp  = nconf_cur = nlos_cur = 0

    def __init__(self):
        super(ACDataEvent, self).__init__(ACDataEventType)


class AMANEvent(QEvent):
    ids = iafs = eats = etas = delays = rwys = spdratios = []

    def __init__(self):
        super(AMANEvent, self).__init__(AMANEventType)


class PanZoomEvent(QEvent):
    def __init__(self, pan=None, zoom=None, origin=None, absolute=False):
        super(PanZoomEvent, self).__init__(PanZoomEventType)
        self.pan      = pan
        self.origin   = origin
        self.zoom     = zoom
        self.absolute = absolute


class SimQuitEvent(QEvent):
    def __init__(self):
        super(SimQuitEvent, self).__init__(SimQuitEventType)
