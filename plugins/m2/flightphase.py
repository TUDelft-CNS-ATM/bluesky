""" This plugin updates the flight phase of all arcraft every update cycle
0 = cruising/hover
1 = climbing 
2 = descending
"""
import numpy as np
import copy
# Import the global bluesky objects. Uncomment the ones you need
from bluesky import core, stack, traf, settings, tools#, navdb, sim, scr, 

def init_plugin():
    ''' Plugin initialisation function. '''
    # Instantiate our example entity
    fp = flightphase()

    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'flightphase',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',
        }

    # init_plugin() should always return a configuration dict.
    return config


class flightphase(core.Entity):
    ''' Example new entity object for BlueSky. '''
    def __init__(self):
        super().__init__()
        with self.settrafarrays():
            self.flightphase = np.array([])
            self.resostrategy = np.array([],dtype='S6') # Name of the resolution method used by each aircraft. added here and used in hybridreso
            self.resoidint = np.array([], dtype=object) # list of intruder callsigns that ownship is curently resolving against. added here and used in hybridreso
            self.resoalt = np.array([]) # array to store the resolution altitude for each aircraft. added here and used in hybridreso
            self.resospd = np.array([]) # array to store the resolution speed for each aircraft. added here and used in hybridreso
            self.resovs = np.array([]) # array to store the resolution vs for each aircraft. added here and used in hybridreso
            self.recoveryspd = np.array([]) # array to store the pre-conflict speed for each aircraft. added here and used in hybridreso
            self.recoveryvs = np.array([]) # array to store the pre-conflict vs for each aircraft. added here and used in hybridreso
            self.resoHdgActive = np.array([], dtype=bool) # asas channels to delete. added here and used in hybridreso
            self.resoTasActive = np.array([], dtype=bool)
            self.resoAltActive = np.array([], dtype=bool)
            self.resoVsActive = np.array([], dtype=bool)
            self.preresoroute = []
        
        # update traf
        traf.flightphase  = self.flightphase
        traf.resostrategy = self.resostrategy
        traf.resoidint = self.resoidint
        traf.resoalt = self.resoalt
        traf.resospd = self.resospd
        traf.resovs = self.resovs
        traf.recoveryspd = self.recoveryspd
        traf.recoveryvs = self.recoveryvs
        traf.resoHdgActive = self.resoHdgActive
        traf.resoTasActive = self.resoTasActive
        traf.resoAltActive = self.resoAltActive
        traf.resoVsActive = self.resoVsActive
        traf.preresoroute = self.preresoroute
        
        # set the vertical speed limit for the cruising aircraft [m/s]
        self.vslimit = 10*tools.aero.fpm
        
        # set the ground speed limit [m/s]
        self.gslimit = 0.5*tools.aero.kts
        

    def create(self, n=1):
        ''' This function gets called automatically when new aircraft are created. '''
        super().create(n)
        self.flightphase[-n:] = 1 # set the initial flight phase to climbing
        self.resostrategy[-n:] = "None" # the initial resolution strategy at creation is "None"
        self.resoidint[-n:] = [[] for i in range(1)]
        self.resoalt[-n:] = -9999
        self.resospd[-n:] = -9999
        self.resovs [-n:] = -9999
        self.recoveryspd[-n:] = -9999
        self.recoveryvs[-n:] = -9999
        self.resoHdgActive[-n:] = False
        self.resoTasActive[-n:] = False
        self.resoAltActive[-n:] = False
        self.resoVsActive[-n:] = False
        self.preresoroute[-n:] = copy.deepcopy(traf.ap.route[-n:])
            
        # update traf
        traf.flightphase = self.flightphase
        traf.resostrategy = self.resostrategy
        traf.resoidint = self.resoidint
        traf.resoalt = self.resoalt
        traf.resospd = self.resospd
        traf.resovs = self.resovs
        traf.recoveryspd = self.recoveryspd
        traf.recoveryvs = self.recoveryvs
        traf.resoHdgActive = self.resoHdgActive
        traf.resoTasActive = self.resoTasActive
        traf.resoAltActive = self.resoAltActive
        traf.resoVsActive = self.resoVsActive
        traf.preresoroute = self.preresoroute
        

    @core.timed_function(name='flightphase', dt=settings.asas_dt/2)
    def update(self):
        ''' Periodically updates the flight phase of all aircraft and their CD look-ahead times '''
        
        # Define the climb and descend conditions. Anything that is not climb or descend is cruise/hover
        # At the moment, it is assumed that cruise / hover are the same in terms of selecting resolutions
        
        climbCondition   = (traf.vs > self.vslimit) * (np.abs(traf.gs) < self.gslimit)
        descendCondition = (traf.vs < -self.vslimit) * (np.abs(traf.gs) < self.gslimit)
        
        # first check for climb. Then descend. Everything else is cruise/hover
        self.flightphase = np.where(climbCondition, 1, np.where(descendCondition, 2, 0))
        
        # set the flightphase into the traffic object so that it can be used in other plugins
        traf.flightphase = self.flightphase
        
        # ------- Adapt the lookahead time for climbing and descending aircraft -------
        
        # Get the max and min vertical speed of ownship 
        vsMinOwn = traf.perf.vsmin
        vsMaxOwn = traf.perf.vsmax
        
        # climbing/descending aircraft should only look up/down one layer at a time
        dtlookup   = np.abs(traf.layerHeight/vsMaxOwn)*1.5
        dtlookdown = np.abs(traf.layerHeight/vsMinOwn)*1.5
        
        # update CD lookahead based on flight phase
        traf.cd.dtlookahead = np.where(climbCondition, dtlookup, np.where(descendCondition, dtlookdown, settings.asas_dtlookahead))
        
        
    @stack.command
    def echoflightphase(self, acid: 'acid'):
        ''' Print the flight phase of the selected aircraft onto the console '''
        flightphase = traf.flightphase[acid]
        
        if flightphase == 0:
            return True, f'{traf.id[acid]} is cruising.'
        elif flightphase == 1:
            return True, f'{traf.id[acid]} is climbing.'
        else:
            return True, f'{traf.id[acid]} is descending.'
        