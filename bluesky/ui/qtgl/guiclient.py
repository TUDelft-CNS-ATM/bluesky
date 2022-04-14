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
from bluesky.core import Signal
from bluesky.tools.aero import ft

# Globals
UPDATE_ALL = ['SHAPE', 'TRAILS', 'CUSTWPT', 'PANZOOM', 'ECHOTEXT', 'ROUTEDATA']
ACTNODE_TOPICS = [b'ACDATA', b'PLOT*', b'ROUTEDATA*']


class GuiClient(Client):
    def __init__(self):
        super().__init__(ACTNODE_TOPICS)
        self.nodedata = dict()
        self.ref_nodedata = nodeData()
        self.discovery_timer = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(20)
        self.subscribe(b'SIMINFO')
        self.subscribe(b'TRAILS')
        self.subscribe(b'PLOT' + self.client_id)
        self.subscribe(b'ROUTEDATA' + self.client_id)

        # Signals
        self.actnodedata_changed = Signal('actnodedata_changed')

    def start_discovery(self):
        super().start_discovery()
        self.discovery_timer = QTimer()
        self.discovery_timer.timeout.connect(self.discovery.send_request)
        self.discovery_timer.start(3000)

    def stop_discovery(self):
        self.discovery_timer.stop()
        self.discovery_timer = None
        super().stop_discovery()

    def stream(self, name, data, sender_id):
        ''' Guiclient stream handler. '''
        changed = ''
        actdata = self.get_nodedata(sender_id)
        if name == b'ACDATA':
            actdata.setacdata(data)
            changed = name.decode('utf8')
        elif name.startswith(b'ROUTEDATA'):
            actdata.setroutedata(data)
            changed = 'ROUTEDATA'
        elif name == b'TRAILS':
            actdata.settrails(**data)
            changed = name.decode('utf8')

        if sender_id == self.act and changed:
            self.actnodedata_changed.emit(sender_id, actdata, changed)

        super().stream(name, data, sender_id)

    def echo(self, text, flags=None, sender_id=None):
        ''' Overloaded Client.echo function. '''
        sender_data = self.get_nodedata(sender_id)
        sender_data.echo(text, flags)
        # If sender_id is None this is an echo command originating from the gui user, and therefore also meant for the active node
        sender_id = sender_id or self.act
        if sender_id == self.act:
            self.actnodedata_changed.emit(sender_id, sender_data, ('ECHOTEXT',))

    def event(self, name, data, sender_id):
        sender_data = self.get_nodedata(sender_id)
        data_changed = []
        if name == b'RESET':
            sender_data.clear_scen_data()
            data_changed = list(UPDATE_ALL)
        elif name == b'SHAPE':
            sender_data.update_poly_data(**data)
            data_changed.append('SHAPE')
        elif name == b'COLOR':
            sender_data.update_color_data(**data)
            if 'polyid' in data:
                data_changed.append('SHAPE')
        elif name == b'DEFWPT':
            sender_data.defwpt(**data)
            data_changed.append('CUSTWPT')
        elif name == b'DISPLAYFLAG':
            sender_data.setflag(**data)
        elif name == b'ECHO':
            
            data_changed.append('ECHOTEXT')
        elif name == b'PANZOOM':
            sender_data.panzoom(**data)
            data_changed.append('PANZOOM')
        elif name == b'SIMSTATE':
            sender_data.siminit(**data)
            data_changed = list(UPDATE_ALL)
        else:
            super().event(name, data, sender_id)

        if sender_id == self.act and data_changed:
            self.actnodedata_changed.emit(sender_id, sender_data, data_changed)

    def actnode_changed(self, newact):
        self.actnodedata_changed.emit(newact, self.get_nodedata(newact), UPDATE_ALL)

    def get_nodedata(self, nodeid=None):
        nodeid = nodeid or self.act
        if not nodeid:
            return self.ref_nodedata

        data = self.nodedata.get(nodeid)
        if not data:
            # If this is a node we haven't addressed yet: create dataset and
            # request node settings
            self.nodedata[nodeid] = data = nodeData()
            self.send_event(b'GETSIMSTATE', target=nodeid)

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

    def settrails(self, swtrails, traillat0, traillon0, traillat1, traillon1):
        if not swtrails:
            self.traillat0 = []
            self.traillon0 = []
            self.traillat1 = []
            self.traillon1 = []
        else:
            self.traillat0.extend(traillat0)
            self.traillon0.extend(traillon0)
            self.traillat1.extend(traillat1)
            self.traillon1.extend(traillon1)

    def clear_scen_data(self):
        # Clear all scenario-specific data for sender node
        self.polys = dict()
        self.custacclr = dict()
        self.custgrclr = dict()
        self.custwplbl = ''
        self.custwplat = np.array([], dtype=np.float32)
        self.custwplon = np.array([], dtype=np.float32)

        self.naircraft = 0
        self.acdata = ACDataEvent()
        self.routedata = RouteDataEvent()

        # Filteralt settings
        self.filteralt = False

        # Create trail data
        self.traillat0 = []
        self.traillon0 = []
        self.traillat1 = []
        self.traillon1 = []

        # Reset transition level
        self.translvl = 4500.*ft

        # Display flags
        self.show_map      = True
        self.show_coast    = True
        self.show_traf     = True
        self.show_pz       = False
        self.show_fir      = True
        self.show_lbl      = 2
        self.show_wpt      = 1
        self.show_apt      = 1
        self.show_poly     = 1  # 0=invisible, 1=outline, 2=fill
        self.ssd_all       = False
        self.ssd_conflicts = False
        self.ssd_ownship   = set()


    def siminit(self, shapes, **kwargs):
        self.__dict__.update(kwargs)
        for shape in shapes:
            self.update_poly_data(**shape)

    def panzoom(self, pan=None, zoom=None, absolute=True):
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

    def defwpt(self, name, lat, lon):
        self.custwplbl += name[:10].ljust(10)
        self.custwplat = np.append(self.custwplat, np.float32(lat))
        self.custwplon = np.append(self.custwplon, np.float32(lon))

    def setflag(self, flag, args=None):
        # Switch/toggle/cycle radar screen features e.g. from SWRAD command
        if flag == 'SYM':
            # For now only toggle PZ
            self.show_pz = not self.show_pz
        # Coastlines
        elif flag == 'GEO':
            self.show_coast = not self.show_coast

        # FIR boundaries
        elif flag == 'FIR':
            self.show_fir = not self.show_fir

        # Airport: 0 = None, 1 = Large, 2= All
        elif flag == 'APT':
            self.show_apt = not self.show_apt

        # Waypoint: 0 = None, 1 = VOR, 2 = also WPT, 3 = Also terminal area wpts
        elif flag == 'VOR' or flag == 'WPT' or flag == 'WP' or flag == 'NAV':
            self.show_wpt = not self.show_wpt

        # Satellite image background on/off
        elif flag == 'SAT':
            self.show_map = not self.show_map

        # Satellite image background on/off
        elif flag == 'TRAF':
            self.show_traf = not self.show_traf

        elif flag == 'POLY':
            self.show_poly = 0 if self.show_poly == 2 else self.show_poly + 1

        elif flag == 'LABEL':
            # Cycle aircraft label through detail level 0,1,2
            if args==None:
                self.show_lbl = (self.show_lbl+1)%3

            # Or use the argument if it is an integer
            else:
                try:
                    self.show_lbl = min(2,max(0,int(args)))
                except:
                    self.show_lbl = (self.show_lbl + 1) % 3

        elif flag == 'SSD':
            self.show_ssd(args)

        elif flag == 'FILTERALT':
            # First argument is an on/off flag
            if args[0]:
                self.filteralt = args[1:]
            else:
                self.filteralt = False

    def echo(self, text='', flags=0):
        if text:
            self.echo_text += ('\n' + text)

    def show_ssd(self, arg):
        if 'ALL' in arg:
            self.ssd_all      = True
            self.ssd_conflicts = False
        elif 'CONFLICTS' in arg:
            self.ssd_all      = False
            self.ssd_conflicts = True
        elif 'OFF' in arg:
            self.ssd_all      = False
            self.ssd_conflicts = False
            self.ssd_ownship = set()
        else:
            remove = self.ssd_ownship.intersection(arg)
            self.ssd_ownship = self.ssd_ownship.union(arg) - remove
