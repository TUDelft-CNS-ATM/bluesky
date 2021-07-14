"""
BlueSky plugin of the M2 Hybrid Team. This plugin will return for each aircraft the type of
airspace layer it is flying in.
Created by: Vincent
Date: 14-07-2021
"""
from random import randint
import numpy as np
# Import the global bluesky objects. Uncomment the ones you need
from bluesky import core, stack, traf, settings, navdb, sim, scr, tools


airspaceStructure =np.genfromtxt('plugins\\m2\\airspace structure spec.csv', delimiter=',',dtype=str)[1:].T

def init_plugin():
    ''' Plugin initialisation function. '''
    # Instantiate our example entity
    airspaceLayers = airspaceLayer()

    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'airspacelayer',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',
        }

    # init_plugin() should always return a configuration dict.
    return config

class airspaceLayer(core.Entity):
    ''' Example new entity object for BlueSky. '''
    def __init__(self):
        super().__init__()
        # All classes deriving from Entity can register lists and numpy arrays
        # that hold per-aircraft data. This way, their size is automatically
        # updated when aircraft are created or deleted in the simulation.
        with self.settrafarrays():
            self.airspacelayertype = np.array([],dtype='S24')

    # def create(self, acid: 'acid', n=1,):
    #     ''' This function gets called automatically when new aircraft are created. '''
    #     # Don't forget to call the base class create when you reimplement this function!
    #     super().create(n)
    #     # After base creation we can change the values in our own states for the new aircraft
    #     self.airspacelayertype[-n:] = [np.where(traf.alt[acid] < 250*0.3048, 'Cruising','Resolving') for _ in range(n)]

    # Functions that need to be called periodically can be indicated to BlueSky
    # with the timed_function decorator
    # def create(self, n=1):
    #     ''' This function gets called automatically when new aircraft are created. '''
    #     # Don't forget to call the base class create when you reimplement this function!
    #     super().create(n)
    #     # After base creation we can change the values in our own states for the new aircraft
    #     self.npassengers[-n:] = [randint(0, 150) for _ in range(n)]




    @core.timed_function(name='example', dt=1)
    def update(self):
        ''' Periodic update function that determines the layer type for all aircraft every second. '''
        for i, callsign in enumerate(traf.id):
            altitude = traf.alt[i]
            layer = airspaceStructure[1][
                np.where(
                    (airspaceStructure[3].astype(int) >= altitude / 0.3048) & (airspaceStructure[2].astype(int) < altitude / 0.3048))]
            self.airspacelayertype[i] = layer[0]
            stack.stack(f'ECHO {callsign} {layer}')
