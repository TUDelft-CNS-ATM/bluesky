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

    @core.timed_function(name='airspacelayer', dt=1)
    def update(self):
        ''' Periodic update function that determines the layer type for all aircraft every second. '''
        for i, callsign in enumerate(traf.id):
            altitude = traf.alt[i]
            layer = airspaceStructure[1][
                np.where(
                    (airspaceStructure[3].astype(int) >= altitude / 0.3048) & (airspaceStructure[2].astype(int) < altitude / 0.3048))]
            self.airspacelayertype[i] = layer[0]
            #stack.stack(f'ECHO {callsign} {layer}') uncomment if you want to keep printing things on the console.
