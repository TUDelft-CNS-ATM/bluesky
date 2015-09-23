try:
    from PyQt4.QtCore import QEvent
except ImportError:
    from PyQt5.QtCore import QEvent

PanZoomEventType, ACDataEventType, SimInfoEventType, StackTextEventType, ShowDialogEventType, DisplayFlagEventType, RouteDataEventType = range(1000, 1007)


class DisplayFlagEvent(QEvent):
    def __init__(self, flag_target='', flag=True):
        super(DisplayFlagEvent, self).__init__(DisplayFlagEventType)
        self.flag_target = flag_target
        self.flag = flag


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
    lat = lon = []
    acidx = 0

    def __init__(self):
        super(RouteDataEvent, self).__init__(RouteDataEventType)


class ACDataEvent(QEvent):
    lat = lon = alt = tas = trk = colors = id = []

    def __init__(self):
        super(ACDataEvent, self).__init__(ACDataEventType)


class PanZoomEvent(QEvent):
    # Internally a panzoom event can be of the following types:
    Pan, PanAbsolute, Zoom, ZoomAbsolute = range(4)

    def __init__(self, pztype, value, origin=None):
        super(PanZoomEvent, self).__init__(PanZoomEventType)
        self.pztype = pztype
        self.value = value
        self.vorigin = origin

    # The corresponding pan lattitude when this is a pan event
    def panlat(self):
        return self.value[0]

    # The corresponding pan longitude when this is a pan event
    def panlon(self):
        return self.value[1]

    # The corresponding value when this is a zoom event
    def zoom(self):
        return self.value

    # Location of the cursor during the event. Used during zoom
    def origin(self):
        return self.vorigin

    # Is this a pan or a zoom event?
    def panzoom_type(self):
        return self.pztype
