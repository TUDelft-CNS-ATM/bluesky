"""
BlueSky plugin of the M2 Hybrid Team. This plugin will return for each aircraft the type of
airspace layer it is flying in. This info is also stored in the traffic object.
Created by: Vincent
Modified by: Emmanuel 
Date: 14-07-2021
"""
from random import randint
import numpy as np
# Import the global bluesky objects. Uncomment the ones you need
from bluesky import core, stack, traf, settings
from bluesky.tools.aero import ft, kts



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
        
        # load the airspace structure 
        self.airspaceStructure = np.genfromtxt('plugins\\m2\\airspace structure spec.csv', delimiter=',',dtype=str, skip_header=2).T
        
        # Process airspace structure and convert to SI units
        self.loweralt = self.airspaceStructure[2].astype(float)*ft  # [m]
        self.upperalt = self.airspaceStructure[3].astype(float)*ft  # [m]
        self.lowerspd = self.airspaceStructure[4].astype(float)*kts # [m/s]
        self.upperspd = self.airspaceStructure[5].astype(float)*kts # [m/s]
        self.layernames = self.airspaceStructure[1]
        
        # add the airspacelayertype as new array per aircraft
        with self.settrafarrays():
            self.airspacelayertype = np.array([],dtype='S24')
        
        # layer height [m]
        traf.layerHeight = abs(self.loweralt[2] - self.loweralt[1])
        
        # set layer variables to traffic so it can be used in other plugins 
        traf.layerLowerAlt = self.loweralt # [m]
        traf.layerUpperAlt = self.upperalt # [m]
        traf.layerLowerSpd = self.lowerspd # [m/s]
        traf.layerUpperSpd = self.upperspd # [m/s]
        traf.layernames    = self.layernames 
        
        # add airspacelayertype to traffic so that it can be used by other plugins and the rest of bluesky
        traf.aclayername = self.airspacelayertype

    @core.timed_function(name='airspacelayer', dt=settings.asas_dt/2)
    def update(self):
        ''' Periodic update function that determines the layer type for all aircraft every second. '''
        
        # Reshape traf.alt to have two dimensions to compare array without looping
        altitudesreshape = traf.alt.reshape(-1, 1)
        
        # compare aircraft altitudes to upper and lower altitude of each layer
        comparelower = self.loweralt <= altitudesreshape
        compareupper = altitudesreshape < self.upperalt
        
        # determine the index of the layer each aircraft is in
        idx = np.where(comparelower & compareupper)[1]
        self.airspacelayertype = self.layernames[idx]
        
        # update the traffic variable
        traf.aclayername = self.airspacelayertype
        
        # Loop through each aircraft and determine which layer it is in currently
        # for i, callsign in enumerate(traf.id):
        #     altitude = traf.alt[i]
        #     layer = self.airspaceStructure[1][
        #         np.where(
        #             (self.upperalt > altitude) & (self.loweralt <= altitude))]
        #     self.airspacelayertype[i] = layer[0]            
        #     stack.stack(f'ECHO {callsign} {layer}') #uncomment if you want to keep printing things on the console.
    
    @stack.command
    def echoaclayer(self, acid: 'acid'):
        ''' Print the layer name of the selected aircraft onto the console '''
        layer = self.getaclayer(acid)
        return True, f'{traf.id[acid]} is in {layer}.'
    
    
    def getaclayer(self, acid: 'acid'):
        ''' Return the name of the layer that the selected aircraft is currently in '''
        layer = self.airspacelayertype[acid]
        return layer
        