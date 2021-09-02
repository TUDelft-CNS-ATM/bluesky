""" BlueSky plugin template. The text you put here will be visible
    in BlueSky as the description of your plugin. """
from random import randint
import numpy as np
# Import the global bluesky objects. Uncomment the ones you need
from bluesky import core, stack, traf, settings, navdb, sim, scr, tools

### Initialization function of your plugin. Do not change the name of this
### function, as it is the way BlueSky recognises this file as a plugin.
def init_plugin():
    ''' Plugin initialisation function. '''
    # Instantiate our example entity
    overshoot = overshootCheck()

    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'overshootcheck',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',
        }

    # init_plugin() should always return a configuration dict.
    return config


class overshootCheck(core.Entity):
    ''' Example new entity object for BlueSky. '''
    def __init__(self):
        super().__init__()
        with self.settrafarrays():
            self.overshot = np.array([], dtype=bool)
            self.wptdist = np.array([])

    def create(self, n=1):
        ''' This function gets called automatically when new aircraft are created. '''
        # Don't forget to call the base class create when you reimplement this function!
        super().create(n)
        # set the initial distance very high so the checker wont trigger when the last waypoing is incidentally very far away
        self.wptdist[-n:] = 99999

    @core.timed_function(name='overshotcheck', dt=5)
    def update(self):
        #import active flightplans
        for i in traf.id:
            idx = traf.id2idx(i)
            dist = self.calc_dist(idx)
            val = self.checker(idx, dist)
            self.overshot[idx] = val


    def calc_dist(self, acid: 'acid'):
        ownship = traf

        #get the current ownship lat lon and flightplan
        ac_lat = ownship.lat[acid]
        ac_lon = ownship.lon[acid]
        ac_route = ownship.ap.route[acid]

        # if the ac does not have a flightplan, argmax will trigger an error
        try:
            #get the index of the last waypoint
            last_wptidx = np.argmax(ac_route.wpname)
        except:
            #if there is no flightplan set the last wptindex to -2 to ensure the next if statement will not get triggered
            last_wptidx = -2

        #if the index of the last waypoint in the flightplan matches with the current active waypoint
        if last_wptidx == ac_route.iactwp:

            # select the lat and lon of the next (and last) waypoint element
            wpt_lat = ac_route.wplat[ac_route.iactwp]
            wpt_lon = ac_route.wplon[ac_route.iactwp]

            #calculate the distance to the next (and last) waypoint
            qdr_to_next, distance_to_wpt = tools.geo.kwikqdrdist(ac_lat, ac_lon, wpt_lat, wpt_lon)
            distance_to_wpt = distance_to_wpt * tools.geo.nm  # Now in meters

            return distance_to_wpt


    def checker(self, acid: 'acid', dist):
        if dist is not None and dist > self.wptdist[acid]:
            val = True
            #TODO Trigger replanning function if True!
        #if distance is calculated and thus the last waypoint is active, set the val still to false and also update the dist in the wptdist array
        elif dist is not None:
            val = False
            self.wptdist[acid] = dist
        elif dist is None:
            val = False
        return val