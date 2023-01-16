''' I/O Client implementation for the QtGL gui. '''
try:
    from PyQt5.QtCore import QTimer
except ImportError:
    from PyQt6.QtCore import QTimer
import numpy as np

from bluesky.ui import palette
from bluesky.ui.polytools import PolygonSet
from bluesky.ui.qtgl.customevents import ACDataEvent, RouteDataEvent
from bluesky.network.client import Client
from bluesky.network import subscriber
from bluesky.core import Signal
from bluesky.tools.aero import ft


# Globals
UPDATE_ALL = ['SHAPE', 'TRAILS', 'CUSTWPT', 'PANZOOM', 'ECHOTEXT', 'ROUTEDATA']


class GuiClient(Client):
    def __init__(self):
        super().__init__()
        self.nodedata = dict()
        self.ref_nodedata = nodeData()
        self.discovery_timer = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(20)

        # Signals
        self.actnodedata_changed = Signal('actnode-changed')

        # Connect to signals. TODO: needs revision
        Signal('SIMSTATE').connect(self.event)
        self.subscribe('RESET').connect(self.event)
        self.subscribe('COLOR').connect(self.event)
        self.subscribe('SHAPE').connect(self.event)
        self.subscribe(b'PANZOOM').connect(self.event)

    def start_discovery(self):
        super().start_discovery()
        self.discovery_timer = QTimer()
        self.discovery_timer.timeout.connect(self.discovery.send_request)
        self.discovery_timer.start(3000)

    def stop_discovery(self):
        self.discovery_timer.stop()
        self.discovery_timer = None
        super().stop_discovery()

    @subscriber
    def echo(self, text='', flags=None, sender_id=b''):
        ''' Overloaded Client.echo function. '''
        # If sender_id is None this is an echo command originating from the gui user, and therefore also meant for the active node
        sender_id = self.sender_id or self.act_id
        sender_data = self.get_nodedata(sender_id)
        sender_data.echo(text, flags)
        if sender_id == self.act_id:
            self.actnodedata_changed.emit(sender_id, sender_data, ('ECHOTEXT',))

    def event(self, *args, **data):
        sender_data = self.get_nodedata(self.sender_id)
        data_changed = []
        if self.topic == b'RESET':
            sender_data.clear_scen_data()
            data_changed = list(UPDATE_ALL)
        elif self.topic == b'SHAPE':
            sender_data.update_poly_data(**data)
            data_changed.append('SHAPE')
        elif self.topic == b'COLOR':
            sender_data.update_color_data(**data)
            if 'polyid' in data:
                data_changed.append('SHAPE')

        elif self.topic == b'PANZOOM':
            sender_data.panzoom(**data)
            data_changed.append('PANZOOM')
        elif self.topic == b'SIMSTATE':
            sender_data.siminit(**data)
            data_changed = list(UPDATE_ALL)
        # else:
        #     super().event(self.topic, data, self.sender_id)

        if self.sender_id == self.act_id and data_changed:
            self.actnodedata_changed.emit(self.sender_id, sender_data, data_changed)

    def actnode_changed(self, newact):
        self.actnodedata_changed.emit(newact, self.get_nodedata(newact), UPDATE_ALL)

    def get_nodedata(self, nodeid=None):
        nodeid = nodeid or self.act_id
        if not nodeid:
            return self.ref_nodedata

        data = self.nodedata.get(nodeid)
        if not data:
            # If this is a node we haven't addressed yet: create dataset and
            # request node settings
            self.nodedata[nodeid] = data = nodeData()
            self.send(b'GETSIMSTATE', to_group=nodeid)

        return data


class nodeData:
    def __init__(self, route=None):
        # Stack window
        self.echo_text = ''
        self.stackcmds = dict()
        self.stacksyn = dict()

        # Display pan and zoom
        self.pan = [0.0, 0.0]
        self.zoom = 1.0

        self.naircraft = 0
        self.acdata = ACDataEvent()
        self.routedata = RouteDataEvent()

        # Per-scenario data
        self.clear_scen_data()

        # Network route to this node
        self._route = route

    def setacdata(self, data):
        self.acdata = ACDataEvent(data)
        self.naircraft = len(self.acdata.lat)

    def setroutedata(self, data):
        self.routedata = RouteDataEvent(data)

    def clear_scen_data(self):
        # Clear all scenario-specific data for sender node
        self.polys = dict()
        self.custacclr = dict()
        self.custgrclr = dict()

        self.naircraft = 0
        self.acdata = ACDataEvent()
        self.routedata = RouteDataEvent()

    def siminit(self, shapes, **kwargs):
        self.__dict__.update(kwargs)
        for shape in shapes:
            self.update_poly_data(**shape)

    def panzoom(self, pan=None, zoom=None, ar=1, absolute=True):
        if pan:
            if absolute:
                self.pan  = list(pan)
            else:
                self.pan[0] += pan[0]
                self.pan[1] += pan[1]
        if zoom:
            self.zoom = zoom * (1.0 if absolute else self.zoom)

    def update_color_data(self, color, acid=None, groupid=None, polyid=None):
        if acid:
            self.custacclr[acid] = tuple(color)
        elif groupid:
            self.custgrclr[groupid] = tuple(color)
        else:
            contourbuf, fillbuf, colorbuf = self.polys.get(polyid)
            color = tuple(color) + (255,)
            colorbuf = np.array(len(contourbuf) // 2 * color, dtype=np.uint8)
            self.polys[polyid] = (contourbuf, fillbuf, colorbuf)

    def update_poly_data(self, name, shape='', coordinates=None, color=None):
        # We're either updating a polygon, or deleting it. In both cases
        # we remove the current one.
        self.polys.pop(name, None)

        # Break up polyline list of (lat,lon)s into separate line segments
        if coordinates is not None:
            if shape == 'LINE' or shape[:4] == 'POLY':
                # Input data is list or array: [lat0,lon0,lat1,lon1,lat2,lon2,lat3,lon3,..]
                newdata = np.array(coordinates, dtype=np.float32)

            elif shape == 'BOX':
                # Convert box coordinates into polyline list
                # BOX: 0 = lat0, 1 = lon0, 2 = lat1, 3 = lon1 , use bounding box
                newdata = np.array([coordinates[0], coordinates[1],
                                 coordinates[0], coordinates[3],
                                 coordinates[2], coordinates[3],
                                 coordinates[2], coordinates[1]], dtype=np.float32)

            elif shape == 'CIRCLE':
                # Input data is latctr,lonctr,radius[nm]
                # Convert circle into polyline list

                # Circle parameters
                Rearth = 6371000.0             # radius of the Earth [m]
                numPoints = 72                 # number of straight line segments that make up the circrle

                # Inputs
                lat0 = coordinates[0]              # latitude of the center of the circle [deg]
                lon0 = coordinates[1]              # longitude of the center of the circle [deg]
                Rcircle = coordinates[2] * 1852.0  # radius of circle [NM]

                # Compute flat Earth correction at the center of the experiment circle
                coslatinv = 1.0 / np.cos(np.deg2rad(lat0))

                # compute the x and y coordinates of the circle
                angles    = np.linspace(0.0, 2.0 * np.pi, numPoints)   # ,endpoint=True) # [rad]

                # Calculate the circle coordinates in lat/lon degrees.
                # Use flat-earth approximation to convert from cartesian to lat/lon.
                latCircle = lat0 + np.rad2deg(Rcircle * np.sin(angles) / Rearth)  # [deg]
                lonCircle = lon0 + np.rad2deg(Rcircle * np.cos(angles) * coslatinv / Rearth)  # [deg]

                # make the data array in the format needed to plot circle
                newdata = np.empty(2 * numPoints, dtype=np.float32)  # Create empty array
                newdata[0::2] = latCircle  # Fill array lat0,lon0,lat1,lon1....
                newdata[1::2] = lonCircle

            # Create polygon contour buffer
            # Distinguish between an open and a closed contour.
            # If this is a closed contour, add the first vertex again at the end
            # and add a fill shape
            if shape[-4:] == 'LINE':
                contourbuf = np.empty(2 * len(newdata) - 4, dtype=np.float32)
                contourbuf[0::4]   = newdata[0:-2:2]  # lat
                contourbuf[1::4]   = newdata[1:-2:2]  # lon
                contourbuf[2::4] = newdata[2::2]  # lat
                contourbuf[3::4] = newdata[3::2]  # lon
                fillbuf = np.array([], dtype=np.float32)
            else:
                contourbuf = np.empty(2 * len(newdata), dtype=np.float32)
                contourbuf[0::4]   = newdata[0::2]  # lat
                contourbuf[1::4]   = newdata[1::2]  # lon
                contourbuf[2:-2:4] = newdata[2::2]  # lat
                contourbuf[3:-3:4] = newdata[3::2]  # lon
                contourbuf[-2:]    = newdata[0:2]
                pset = PolygonSet()
                pset.addContour(newdata)
                fillbuf = np.array(pset.vbuf, dtype=np.float32)

            # Define color buffer for outline
            defclr = tuple(color or palette.polys) + (255,)
            colorbuf = np.array(len(contourbuf) // 2 * defclr, dtype=np.uint8)

            # Store new or updated polygon by name, and concatenated with the
            # other polys
            self.polys[name] = (contourbuf, fillbuf, colorbuf)

    def echo(self, text='', flags=0, sender_id=None):
        if text:
            self.echo_text += ('\n' + text)
