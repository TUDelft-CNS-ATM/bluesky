""" BlueSky plugin template. The text you put here will be visible
    in BlueSky as the description of your plugin. """
from random import randint
import pandas as pd
import numpy as np
# Import the global bluesky objects. Uncomment the ones you need
from bluesky import core, stack, traf  #, settings, navdb, sim, scr, tools

### Initialization function of your plugin. Do not change the name of this
### function, as it is the way BlueSky recognises this file as a plugin.
def init_plugin():
    ''' Plugin initialisation function. '''
    # Instantiate our example entity
    example = DF_arrays()

    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'DFFUN',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',
        }

    # init_plugin() should always return a configuration dict.
    return config


class DF_arrays(core.Entity):
    ''' Example new entity object for BlueSky. '''
    def __init__(self):
        super().__init__()

        with self.settrafarrays():
            self.npassengers = pd.DataFrame({
                                            'a': pd.Series(dtype='int'),
                                            'b': pd.Series(dtype='str'),
                                            'c': pd.Series(dtype='float')
                                            })

    def create(self, n=1):
        ''' This function gets called automatically when new aircraft are created. '''
        # Don't forget to call the base class create when you reimplement this function!
        super().create(n)
        # After base creation we can change the values in our own states for the new aircraft

        self.npassengers.loc[-n:, 'a'] = 5
        self.npassengers.loc[-n:, 'b'] = 'test'
        self.npassengers.loc[-n:, 'c'] = 10.0