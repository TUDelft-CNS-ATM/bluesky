try:
    from PyQt4.QtCore import QEvent
except ImportError:
    from PyQt5.QtCore import QEvent


class ACDataEvent(object):
    lat = lon = alt = tas = trk = colors = ids = []


class PanZoomEvent(QEvent):
    PanZoomEventType, Pan, PanAbsolute, Zoom = range(4)

    def __init__(self, pztype, value, origin=None):
        super(PanZoomEvent, self).__init__(self.PanZoomEventType)
        self.pztype = pztype
        self.value = value
        self.vorigin = origin

    # The corresponding value when this is a pan event
    def pan(self):
        return self.value

    # The corresponding value when this is a zoom event
    def zoom(self):
        return self.value

    # Location of the cursor during the event. Used during zoom
    def origin(self):
        return self.vorigin

    # Is this a pan or a zoom event?
    def panzoom_type(self):
        return self.pztype
