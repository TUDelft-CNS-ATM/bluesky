""" ScreenIO is a screen proxy on the simulation side for the QTGL implementation of BlueSky."""
import time
import numpy as np

# Local imports
import bluesky as bs
from bluesky import stack
from bluesky.tools import Timer, areafilter


class ScreenIO(object):
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
    def __init__(self):
        # Screen state defaults
        self.def_pan     = (0.0, 0.0)
        self.def_zoom    = 1.0
        # Screen state overrides per client
        self.client_pan  = dict()
        self.client_zoom = dict()
        self.client_ar   = dict()
        self.route_acid  = dict()

        # Dicts of custom aircraft and group colors
        self.custacclr = dict()
        self.custgrclr = dict()

        # Timing bookkeeping counters
        self.prevtime    = 0.0
        self.samplecount = 0
        self.prevcount   = 0

        # Output event timers
        self.slow_timer = Timer()
        self.slow_timer.timeout.connect(self.send_siminfo)
        self.slow_timer.timeout.connect(self.send_route_data)
        self.slow_timer.start(int(1000 / self.siminfo_rate))

        self.fast_timer = Timer()
        self.fast_timer.timeout.connect(self.send_aircraft_data)
        self.fast_timer.start(int(1000 / self.acupdate_rate))

    def update(self):
        if bs.sim.state == bs.OP:
            self.samplecount += 1

    def reset(self):
        self.custacclr = dict()
        self.custgrclr = dict()
        self.samplecount = 0
        self.prevcount   = 0
        self.prevtime    = 0.0

        # Communicate reset to gui
        bs.sim.send_event(b'RESET', b'ALL')


    def echo(self, text='', flags=0):
        bs.sim.send_event(b'ECHO', dict(text=text, flags=flags))

    def cmdline(self, text):
        bs.sim.send_event(b'CMDLINE', text)

    def getviewctr(self):
        return self.client_pan.get(stack.sender()) or self.def_pan

    def getviewbounds(self):
        # Get appropriate lat/lon/zoom/aspect ratio
        sender   = stack.sender()
        lat, lon = self.client_pan.get(sender) or self.def_pan
        zoom     = self.client_zoom.get(sender) or self.def_zoom
        ar       = self.client_ar.get(sender) or 1.0

        lat0 = lat - 1.0 / (zoom * ar)
        lat1 = lat + 1.0 / (zoom * ar)
        lon0 = lon - 1.0 / (zoom * np.cos(np.radians(lat)))
        lon1 = lon + 1.0 / (zoom * np.cos(np.radians(lat)))
        return lat0, lat1, lon0, lon1

    def zoom(self, zoom, absolute=True):
        sender    = stack.sender()
        if sender:
            if absolute:
                self.client_zoom[sender] = zoom
            else:
                self.client_zoom[sender] = zoom * self.client_zoom.get(sender, self.def_zoom)
        else:
            self.def_zoom = zoom * (1.0 if absolute else self.def_zoom)
            self.client_zoom.clear()

        bs.sim.send_event(b'PANZOOM', dict(zoom=zoom, absolute=absolute))

    def color(self, name, r, g, b):
        ''' Set custom color for aircraft or shape. '''
        data = dict(color=(r, g, b))
        if name in bs.traf.groups:
            groupmask = bs.traf.groups.groups[name]
            data['groupid'] = groupmask
            self.custgrclr[groupmask] = (r, g, b)
        elif name in bs.traf.id:
            data['acid'] = name
            self.custacclr[name] = (r, g, b)
        elif areafilter.hasArea(name):
            data['polyid'] = name
            areafilter.areas[name].raw['color'] = (r, g, b)
        else:
            return False, 'No object found with name ' + name
        bs.sim.send_event(b'COLOR', data)
        return True

    def pan(self, *args):
        ''' Move center of display, relative of to absolute position lat,lon '''
        lat, lon = 0, 0
        absolute = False
        if args[0] == "LEFT":
            lon = -0.5
        elif args[0] == "RIGHT":
            lon = 0.5
        elif args[0] == "UP":
            lat = 0.5
        elif args[0] == "DOWN":
            lat = -0.5
        else:
            absolute = True
            lat, lon = args

        sender    = stack.sender()
        if sender:
            if absolute:
                self.client_pan[sender] = (lat, lon)
            else:
                ll = self.client_pan.get(sender) or self.def_pan
                self.client_pan[sender] = (lat + ll[0], lon + ll[1])
        else:
            self.def_pan = (lat,lon) if absolute else (lat + self.def_pan[0],
                                                       lon + self.def_pan[1])
            self.client_pan.clear()

        bs.sim.send_event(b'PANZOOM', dict(pan=(lat,lon), absolute=absolute))

    def shownd(self, acid):
        bs.sim.send_event(b'SHOWND', acid)

    def symbol(self):
        bs.sim.send_event(b'DISPLAYFLAG', dict(flag='SYM'))

    def feature(self, switch, argument=None):
        bs.sim.send_event(b'DISPLAYFLAG', dict(flag=switch, args=argument))

    def trails(self,sw):
        bs.sim.send_event(b'DISPLAYFLAG', dict(flag='TRAIL', args=sw))

    def showroute(self, acid):
        ''' Toggle show route for this aircraft '''
        self.route_acid[stack.sender()] = acid
        return True

    def addnavwpt(self, name, lat, lon):
        ''' Add custom waypoint to visualization '''
        bs.sim.send_event(b'DEFWPT', dict(name=name, lat=lat, lon=lon))
        return True

    def show_file_dialog(self):
        bs.sim.send_event(b'SHOWDIALOG', dict(dialog='OPENFILE'))
        return ''

    def show_cmd_doc(self, cmd=''):
        bs.sim.send_event(b'SHOWDIALOG', dict(dialog='DOC', args=cmd))

    def filteralt(self, *args):
        bs.sim.send_event(b'DISPLAYFLAG', dict(flag='FILTERALT', args=args))

    def objappend(self, objtype, objname, data):
        """Add a drawing object to the radar screen using the following inputs:
           objtype: "LINE"/"POLY" /"BOX"/"CIRCLE" = string with type of object
           objname: string with a name as key for reference
           objdata: lat,lon data, depending on type:
                    POLY/LINE: lat0,lon0,lat1,lon1,lat2,lon2,....
                    BOX : lat0,lon0,lat1,lon1   (bounding box coordinates)
                    CIRCLE: latctr,lonctr,radiusnm  (circle parameters)
        """
        bs.sim.send_event(b'SHAPE', dict(name=objname, shape=objtype, coordinates=data))

    def event(self, eventname, eventdata, sender_rte):
        if eventname == b'PANZOOM':
            self.client_pan[sender_rte[-1]]  = eventdata['pan']
            self.client_zoom[sender_rte[-1]] = eventdata['zoom']
            self.client_ar[sender_rte[-1]]   = eventdata['ar']
            return True

        return False

    # =========================================================================
    # Slots
    # =========================================================================
    def send_siminfo(self):
        t  = time.time()
        dt = np.maximum(t - self.prevtime, 0.00001)  # avoid divide by 0
        speed = (self.samplecount - self.prevcount) / dt * bs.sim.simdt
        bs.sim.send_stream(b'SIMINFO', (speed, bs.sim.simdt, bs.sim.simt,
            str(bs.sim.utc.replace(microsecond=0)), bs.traf.ntraf, bs.sim.state, stack.get_scenname()))
        self.prevtime  = t
        self.prevcount = self.samplecount

    def send_aircraft_data(self):
        data = dict()
        data['simt']       = bs.sim.simt
        data['id']         = bs.traf.id
        data['lat']        = bs.traf.lat
        data['lon']        = bs.traf.lon
        data['alt']        = bs.traf.alt
        data['tas']        = bs.traf.tas
        data['cas']        = bs.traf.cas
        data['gs']         = bs.traf.gs
        data['ingroup']    = bs.traf.groups.ingroup
        data['inconf'] = bs.traf.asas.inconf
        data['tcpamax'] = bs.traf.asas.tcpamax
        data['nconf_cur'] = len(bs.traf.asas.confpairs_unique)
        data['nconf_tot'] = len(bs.traf.asas.confpairs_all)
        data['nlos_cur'] = len(bs.traf.asas.lospairs_unique)
        data['nlos_tot'] = len(bs.traf.asas.lospairs_all)
        data['trk']        = bs.traf.trk
        data['vs']         = bs.traf.vs
        data['vmin']       = bs.traf.asas.vmin
        data['vmax']       = bs.traf.asas.vmax

        # Trails, send only new line segments to be added
        data['swtrails']  = bs.traf.trails.active
        data['traillat0'] = bs.traf.trails.newlat0
        data['traillon0'] = bs.traf.trails.newlon0
        data['traillat1'] = bs.traf.trails.newlat1
        data['traillon1'] = bs.traf.trails.newlon1
        bs.traf.trails.clearnew()

        # Last segment which is being built per aircraft
        data['traillastlat']   = bs.traf.trails.lastlat
        data['traillastlon']   = bs.traf.trails.lastlon

        # Transition level as defined in traf
        data['translvl']   = bs.traf.translvl

        # ASAS resolutions for visualization. Only send when evaluated
        if bs.traf.asas.asaseval:
            data['asasn']  = bs.traf.asas.asasn
            data['asase']  = bs.traf.asas.asase
        else:
            data['asasn']  = np.zeros(bs.traf.ntraf, dtype=np.float32)
            data['asase']  = np.zeros(bs.traf.ntraf, dtype=np.float32)

        bs.sim.send_stream(b'ACDATA', data)

    def send_route_data(self):
        for sender, acid in self.route_acid.items():
            data               = dict()
            data['acid']       = acid
            idx   = bs.traf.id2idx(acid)
            if idx >= 0:
                route          = bs.traf.ap.route[idx]
                data['iactwp'] = route.iactwp

                # We also need the corresponding aircraft position
                data['aclat']  = bs.traf.lat[idx]
                data['aclon']  = bs.traf.lon[idx]

                data['wplat']  = route.wplat
                data['wplon']  = route.wplon

                data['wpalt']  = route.wpalt
                data['wpspd']  = route.wpspd

                data['wpname'] = route.wpname

            bs.sim.send_stream(b'ROUTEDATA' + (sender or b'*'), data)  # Send route data to GUI
