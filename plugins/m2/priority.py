""" This plugin can be used to set the priority of an aircraft
1 = low priority
2 = medium priority 
3 = high priority 
"""
import numpy as np
# Import the global bluesky objects. Uncomment the ones you need
from bluesky import core, stack, traf, settings #, navdb, sim, scr, tools

### Initialization function of your plugin. Do not change the name of this
### function, as it is the way BlueSky recognises this file as a plugin.
def init_plugin():
    ''' Plugin initialisation function. '''
    # Instantiate our example entity
    prio = priority()

    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'priority',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',
        }

    # init_plugin() should always return a configuration dict.
    return config


class priority(core.Entity):
    ''' Example new entity object for BlueSky. '''
    def __init__(self):
        super().__init__()
        with self.settrafarrays():
            self.priority = np.array([]) # set the default priority to 1 (low priority)
        traf.priority = self.priority
        
    
    def create(self, n=1):
        ''' This function sets the default priority for all created aircraft 
        to 1 (low priority) '''
        super().create(n)
        self.priority[-n:] = 1
        traf.priority = self.priority
        
        
    @stack.command
    def setpriority(self, acid: 'acid', prio):
        ''' Set the priority of 'acid' to 'prio'. '''
        self.priority[acid] = prio
        traf.priority = self.priority
        return True, f'The priority of {traf.id[acid]} is set to {prio}.'
    
    @stack.command
    def getpriority(self, acid: 'acid'):
        ''' Print the priority of 'acid'. '''
        return True, f'The priority of {traf.id[acid]} is currently {self.priority[acid]}.'
