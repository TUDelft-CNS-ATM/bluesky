""" BlueSky sector occupancy count plugin. """
import numpy as np
# Import the global bluesky objects. Uncomment the ones you need
from bluesky import traf, scr  #, stack, settings, navdb, traf, sim, scr, tools
from bluesky.tools import areafilter, datalog

# List of sectors known to this plugin.
sectors    = list()
# List of aircraft counted inside registered sectors in previous update step.
previnside = list()

# Data logger for sector occupancy count logfiles
logger     = None

### Initialization function of your plugin. Do not change the name of this
### function, as it is the way BlueSky recognises this file as a plugin.
def init_plugin():
    # Register a sector count logger
    global logger
    logger = datalog.crelog('OCCUPANCYLOG', None, 'Sector count log')

    # Configuration parameters
    config = {
        'plugin_name':     'SECTORCOUNT',
        'plugin_type':     'sim',
        'update_interval': 3.0,
        'update':          update
        }

    stackfunctions = {
        # The command name for your function
        'SECTORCOUNT': [
            # A short usage string. This will be printed if you type HELP <name> in the BlueSky console
            'SECTORCOUNT LIST OR ADD sectorname or REMOVE sectorname',

            # A list of the argument types your function accepts. For a description of this, see ...
            'txt,[txt]',

            # The name of your function in this plugin
            sectorcount,

            # a longer help text of your function.
            'Add/remove/list sectors for occupancy count']
    }

    # init_plugin() should always return these two dicts.
    return config, stackfunctions


### Periodic update functions that are called by the simulation. You can replace
### this by anything, so long as you communicate this in init_plugin

def update():
    mylog = list()
    for idx, name in enumerate(sectors):
        # Perform inside count, and check entering and leaving aircraft
        inside    = areafilter.checkInside(name, traf.lat, traf.lon, traf.alt)
        ids       = set(np.array(traf.id)[inside])
        previds   = previnside[idx]
        arrived   = str.join(', ', ids - previds)
        left      = str.join(', ', previds - ids)
        n_tot     = len(ids)
        n_arrived = len(arrived)
        n_left    = len(left)

        # Add log string to list
        mylog.append('%s, %d' % (name, n_tot))

        # Print to console
        if n_left > 0:
            scr.echo('%s aircraft that have left: %s' % (name, left))
        if n_arrived > 0:
            scr.echo('%s aircraft that have arrived: %s' % (name, arrived))
        if n_left + n_arrived > 0:
            scr.echo('%s occupancy count: %d' % (name, n_tot))
            previnside[idx] = ids

    # Log data if enabled
    logger.log(str.join(', ', mylog))

def sectorcount(sw, name=''):
    if sw == 'LIST':
        if len(sectors) == 0:
            return True, 'No registered sectors available'
        else:
            return True, 'Registered sectors:', str.join(', ', sectors)
    elif sw == 'ADD':
        # Add new sector to list.
        if name in sectors:
            return True, 'Sector %s already registered.' % name
        elif areafilter.hasArea(name):
            # Add new area to the sector list, and add an initial inside count of traffic
            sectors.append(name)
            inside = areafilter.checkInside(name, traf.lat, traf.lon, traf.alt)
            previnside.append(set(np.array(traf.id)[inside]))
            return True, 'Added %s to sector list.' % name
        else:
            return False, "No area found with name '%s', create it first with one of the shape commands" % name

    else:
        # Remove area from sector list
        if name in sectors:
            idx = sectors.index(name)
            sectors.pop(idx)
            previnside.pop(idx)
            return True, 'Removed %s from sector list.' % name
        else:
            return False, "No sector registered with name '%s'." % name
