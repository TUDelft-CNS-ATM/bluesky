try:
    from PyQt5.QtCore import QEvent
except ImportError:
    from PyQt4.QtCore import QEvent

PanZoomEventType, ACDataEventType, SimInfoEventType, StackTextEventType,  \
ShowDialogEventType, DisplayFlagEventType, RouteDataEventType,  \
DisplayShapeEventType = range(1000, 1008)


class DisplayFlagEvent(QEvent):
    def __init__(self, switch='', argument=''):
        super(DisplayFlagEvent, self).__init__(DisplayFlagEventType)
        self.switch = switch
        self.argument = argument


class SimInfoEvent(QEvent):
    def __init__(self, sys_freq, simdt, simt, n_ac, mode):
        super(SimInfoEvent, self).__init__(SimInfoEventType)
        self.sys_freq = sys_freq
        self.simdt    = simdt
        self.simt     = simt
        self.n_ac     = n_ac
        self.mode     = mode


class StackTextEvent(QEvent):
    def __init__(self, text):
        super(StackTextEvent, self).__init__(StackTextEventType)
        self.text = text


class ShowDialogEvent(QEvent):
    # Types of dialog
    filedialog_type = 0

    def __init__(self, dialog_type=0):
        super(ShowDialogEvent, self).__init__(ShowDialogEventType)
        self.dialog_type = dialog_type


class RouteDataEvent(QEvent):
    lat = lon = wptlabels = []
    acidx = 0

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

    def __init__(self):
        super(ACDataEvent, self).__init__(ACDataEventType)


class PanZoomEvent(QEvent):
    def __init__(self, pan=None, zoom=None, origin=None, absolute=False):
        super(PanZoomEvent, self).__init__(PanZoomEventType)
        self.pan      = pan
        self.origin   = origin
        self.zoom     = zoom
        self.absolute = absolute
