""" This plugin updates the flight phase of all arcraft every update cycle
0 = cruising
1 = climbing
2 = descending
"""
import numpy as np
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
        traf.flightphase = self.flightphase
        
        # set the vertical speed limit for the cruising aircraft
        self.vslimit = 10*tools.aero.fpm

    def create(self, n=1):
        ''' This function gets called automatically when new aircraft are created. '''
        super().create(n)
        self.flightphase[-n:] = 1 # set the initial flight phase to climbing
        traf.flightphase = self.flightphase

    @core.timed_function(name='flightphase', dt=settings.asas_dt)
    def update(self):
        ''' Periodically updates the flight phase of all aircraft '''
        
        # use np.where to figure out the flight phase of each aircaft with 
        # self.limit as the decision criteria (output is array with 0 = cruising etc.)
        self.flightphase = np.where(np.abs(traf.vs)<=self.vslimit, 0, np.where(traf.vs>self.vslimit,1,2))
        
        # set the flightphase into the traffic object so that it can be used in other plugins
        traf.flightphase = self.flightphase
        
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
        