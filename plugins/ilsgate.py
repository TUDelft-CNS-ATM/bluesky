""" BlueSky plugin template. The text you put here will be visible
    in BlueSky as the description of your plugin. """
import numpy as np
# Import the global bluesky objects. Uncomment the ones you need
from bluesky import stack  #, settings, navdb, traf, sim, scr, tools
from bluesky import navdb
from bluesky.tools.aero import ft
from bluesky.tools import geo, areafilter

### Initialization function of your plugin. Do not change the name of this
### function, as it is the way BlueSky recognises this file as a plugin.
def init_plugin():

    # Addtional initilisation code

    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'ILSGATE',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',

        # Update interval in seconds. By default, your plugin's update function(s)
        # are called every timestep of the simulation. If your plugin needs less
        # frequent updates provide an update interval.
        'update_interval': 0.0,

        # The update function is called after traffic is updated. Use this if you
        # want to do things as a result of what happens in traffic. If you need to
        # something before traffic is updated please use preupdate.
        'update':          update,

        # If your plugin has a state, you will probably need a reset function to
        # clear the state in between simulations.
        'reset':         reset
        }

    stackfunctions = {
        # The command name for your function
        'ILSGATE': [
            # A short usage string. This will be printed if you type HELP <name> in the BlueSky console
            'ILSGATE Airport/runway',

            # A list of the argument types your function accepts. For a description of this, see ...
            'txt',

            # The name of your function in this plugin
            ilsgate,

            # a longer help text of your function.
            'Define an ILS approach area for a given runway.']
    }

    # init_plugin() should always return these two dicts.
    return config, stackfunctions


### Periodic update functions that are called by the simulation. You can replace
### this by anything, so long as you communicate this in init_plugin

def update():
    pass

def reset():
    pass

### Other functions of your plugin
def ilsgate(rwyname):
    if '/' not in rwyname:
        return False, 'Argument is not a runway ' + rwyname
    apt, rwy = rwyname.split('/RW')
    rwy = rwy.lstrip('Y')
    apt_thresholds = navdb.rwythresholds.get(apt)
    if not apt_thresholds:
        return False, 'Argument is not a runway (airport not found) ' + apt
    rwy_threshold = apt_thresholds.get(rwy)
    if not rwy_threshold:
        return False, 'Argument is not a runway (runway not found) ' + rwy
    # Extract runway threshold lat/lon, and runway heading
    lat, lon, hdg = rwy_threshold

    # The ILS gate is defined as a triangular area pointed away from the runway
    # First calculate the two far corners in cartesian coordinates
    cone_length = 50 # nautical miles
    cone_angle  = 20.0 # degrees
    lat1, lon1 = geo.qdrpos(lat, lon, hdg - 180.0 + cone_angle, cone_length)
    lat2, lon2 = geo.qdrpos(lat, lon, hdg - 180.0 - cone_angle, cone_length)
    coordinates = np.array([lat, lon, lat1, lon1, lat2, lon2])
    areafilter.defineArea('ILS' + rwyname, 'POLYALT', coordinates, top=4000*ft)
