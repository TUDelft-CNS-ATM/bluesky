""" This plugin manages all the datalogs for the USEPE project. """
# Import the global bluesky objects. Uncomment the ones you need
from bluesky import core, traf  #, stack, settings, navdb, sim, scr, tools
from bluesky.tools import datalog

# Datalog for all conflicts
conflog = None
loslog = None

# Parameters used when logging
confheader = \
    'CONFLICTS LOG\n' + \
    'Start and end of all conflicts\n\n' + \
    'Simulation Time [s], UAS1, UAS2, Start/End'

losheader = \
    'LOSS OF SEPARATION LOG\n' + \
    'Start and end of all loss of separation\n\n' + \
    'Simulation Time [s], UAS1, UAS2, Start/End'

### Initialisation function of your plugin. Do not change the name of this
### function, as it is the way BlueSky recognises this file as a plugin.
def init_plugin():
    ''' Plugin initialisation function. '''
    # Instantiate the UsepeLogger entity
    usepelogger = UsepeLogger()

    global conflog
    global loslog
    conflog = datalog.crelog('USEPECONFLOG', None, confheader)
    loslog = datalog.crelog('USEPELOSLOG', None, losheader)

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
        
        # These lists stores the events from the previous step, 
        # ensuring each event is logged only once and that we know when they have ended.
        self.prevconf = list()
        self.prevlos = list()

    def update(self):
        self.los_logger()

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
    
    def los_logger(self):
        currentlos = list()

        # Go through all loss of separation pairs and sort the IDs for easier matching
        currentlos = [list(pair) for pair in traf.cd.lospairs_unique]
        for pair in currentlos: pair.sort()
        
        # Create lists of all new and ended LoS
        startlos = [currpair for currpair in currentlos if currpair not in self.prevlos]
        endlos = [prevpair for prevpair in self.prevlos if prevpair not in currentlos]

        # Log start and end of LoS
        loslog.log(startlos, 'start')
        loslog.log(endlos, 'end')

        # for pair in traf.cd.lospairs_unique:
        #     uas1, uas2 = pair
        #     sortedpair = [uas1, uas2]
        #     sortedpair.sort()
        #     currentlos.append(sortedpair)
        #     if sortedpair not in self.prevlos:
        #         loslog.log(f' {sortedpair[0]}, {sortedpair[1]}, start')

        # # Log all ended loss of separation
        # for pair in self.prevlos:
        #     if pair not in currentlos:
        #         loslog.log(f' {pair[0]}, {pair[1]}, end')
        
        # Store the new loss of separation environment
        self.prevlos = currentlos
