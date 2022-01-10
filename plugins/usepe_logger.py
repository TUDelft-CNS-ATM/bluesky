""" This plugin manages all the datalogs for the USEPE project. """
# Import the global bluesky objects. Uncomment the ones you need
from bluesky import core, traf  #, stack, settings, navdb, sim, scr, tools
from bluesky.tools import datalog

# Datalog for all conflicts
conflog = None

# Parameters used when logging
confheader = \
    'CONFLICTS LOG\n' + \
    'Beginning and end of all conflicts\n\n' + \
    'Simulation Time [s], UAS1, UAS2, Start/End'

### Initialisation function of your plugin. Do not change the name of this
### function, as it is the way BlueSky recognises this file as a plugin.
def init_plugin():
    ''' Plugin initialisation function. '''
    # Instantiate the UsepeLogger entity
    usepelogger = UsepeLogger()

    global conflog
    conflog = datalog.crelog('USEPECONFLOG', None, confheader)

    # Configuration parameters
    config = {
        'plugin_name':     'USEPELOGGER',
        'plugin_type':     'sim',
        'update_interval': 5.0,
        'update': usepelogger.update
        }

    # init_plugin() should always return a configuration dict.
    return config

class UsepeLogger(core.Entity):
    ''' Provides the needed funcionality for each log '''

    def __init__(self):
        super().__init__()
        
        # This list stores the conflicts from the previous step, 
        # ensuring each conflict is logged only once and that we know when they have ended.
        self.prevconf = list()

    def update(self):
        currentconf = list()
        
        # Go through all conflict pairs and sort the IDs for easier matching
        # Log any new conflict
        for pair in traf.cd.confpairs_unique:
            uas1, uas2 = pair
            sortedpair = [uas1, uas2]
            sortedpair.sort()
            currentconf.append(sortedpair)
            if sortedpair not in self.prevconf:
                conflog.log(f' {sortedpair[0]}, {sortedpair[1]}, start')
        
        # Log all ended conflicts
        for pair in self.prevconf:
            if pair not in currentconf:
                conflog.log(f' {pair[0]}, {pair[1]}, end')

        # Store the new conflict environment
        self.prevconf = currentconf
