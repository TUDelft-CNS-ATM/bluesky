""" BlueSky flight statistics logger plugin. """
import numpy as np
# Import the global bluesky objects. Uncomment the ones you need
from bluesky import traf, sim  #, settings, navdb, traf, sim, scr, tools
from bluesky.tools import datalog, TrafficArrays, RegisterElementParameters

header = \
    "#######################################################\n" + \
    "FLST LOG\n" + \
    "Flight Statistics\n" + \
    "#######################################################\n\n" + \
    "Parameters [Units]:\n" + \
    "Deletion Time [s], " + \
    "Call sign [-], " + \
    "Spawn Time [s], " + \
    "Flight time [s], " + \
    "Actual Distance 2D [m], " + \
    "Actual Distance 3D [m], " + \
    "Work Done [J], " + \
    "Latitude [deg], " + \
    "Longitude [deg], " + \
    "Altitude [m], " + \
    "TAS [m/s], " + \
    "Vertical Speed [m/s], " + \
    "Heading [deg], " + \
    "Origin Lat [deg], " + \
    "Origin Lon [deg], " + \
    "Destination Lat [deg], " + \
    "Destination Lon [deg], " + \
    "ASAS Active [bool], " + \
    "Pilot ALT [m], " + \
    "Pilot SPD (TAS) [m/s], " + \
    "Pilot HDG [deg], " + \
    "Pilot VS [m/s]"  + "\n"

# Global data
flstlogger = None

### Initialization function of your plugin. Do not change the name of this
### function, as it is the way BlueSky recognises this file as a plugin.
def init_plugin():

    # Addtional initilisation code
    global flstlogger
    flstlogger = FLST()

    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'FLSTLOG',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',

        # Update interval in seconds. By default, your plugin's update function(s)
        # are called every timestep of the simulation. If your plugin needs less
        # frequent updates provide an update interval.
        'update_interval': 2.5,

        # The update function is called after traffic is updated. Use this if you
        # want to do things as a result of what happens in traffic. If you need to
        # something before traffic is updated please use preupdate.
        'update':          flstlogger.update,
        }

    # init_plugin() should always return these two dicts.
    return config, {}

class FLST(TrafficArrays):
    def __init__(self):
        super(FLST, self).__init__()
        self.logger = datalog.defineLogger('FLSTLOG', header)

        with RegisterElementParameters(self):
            self.distance2D  = np.array([])
            self.distance3D  = np.array([])
            self.work        = np.array([])
            self.create_time = np.array([])

    def create(self, n=1):
        super(FLST, self).create(n)
        self.create_time[-n:] = sim.simt

    def update(self):
        # Update flight efficiency metrics
        # 2D and 3D distance [m], and work done (force*distance) [J]
        resultantspd = np.sqrt(traf.gs * traf.gs + traf.vs * traf.vs)
        self.distance2D += sim.simdt * traf.gs
        self.distance3D += sim.simdt * resultantspd
        self.work += (traf.perf.Thr * sim.simdt * resultantspd)

        # Determine deleted aircraft and log if aircraft were deleted
        delidx = traf.area.delidx
        if delidx:
            self.logger.log(
                np.array(traf.id)[delidx],
                self.create_time[delidx],
                sim.simt - self.create_time[delidx],
                self.distance2D[delidx],
                self.distance3D[delidx],
                self.work[delidx],
                traf.lat[delidx],
                traf.lon[delidx],
                traf.alt[delidx],
                traf.tas[delidx],
                traf.vs[delidx],
                traf.hdg[delidx],
                traf.ap.origlat[delidx],
                traf.ap.origlon[delidx],
                traf.ap.destlat[delidx],
                traf.ap.destlon[delidx],
                traf.asas.active[delidx],
                traf.pilot.alt[delidx],
                traf.pilot.spd[delidx],
                traf.pilot.vs[delidx],
                traf.pilot.hdg[delidx]
            )
