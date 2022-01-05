""" To-Do """
# Import the global bluesky objects. Uncomment the ones you need
from bluesky import core, traf  #, stack, settings, navdb, sim, scr, tools
from bluesky.tools import datalog

logger = None

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

    global logger
    logger = datalog.crelog('USEPECONFLOG', None, confheader)

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
    ''' To-Do '''

    def __init__(self):
        super().__init__()
        
        self.prevconf = list()

    def update(self):
        currentconf = list()
        for pair in traf.cd.confpairs_unique:
            uas1, uas2 = pair
            sortedpair = [uas1, uas2]
            sortedpair.sort()
            currentconf.append(sortedpair)
            if sortedpair not in self.prevconf:
                logger.log(f' {sortedpair[0]}, {sortedpair[1]}, start')
        
        for pair in self.prevconf:
            if pair not in currentconf:
                logger.log(f' {pair[0]}, {pair[1]}, end')
        
        self.prevconf = currentconf
