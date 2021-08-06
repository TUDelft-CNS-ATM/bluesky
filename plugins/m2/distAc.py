""" The distac plugin has a stack command that echos the distance and bearing 
    between two aircraft given their callsigns.
    Created by: Emmanuel 
    Date: 22 July 2021 
    """
from random import randint
import numpy as np
# Import the global bluesky objects. Uncomment the ones you need
from bluesky import core, stack, traf, settings, tools

def init_plugin():
    ''' Plugin initialisation function. '''
    # Instantiate our example entity
    distac = distAc()

    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'distAc',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',
        }

    # init_plugin() should always return a configuration dict.
    return config


class distAc(core.Entity):
    ''' Example new entity object for BlueSky. '''
    def __init__(self):
        super().__init__()

    
    def distAc(self, acid1: 'acid', acid2: 'acid'):
        ''' Calculate the distance between two selected aircraft on the console'''
        
        # lat and lon and hdg of acid1
        lat1 = traf.lat[acid1]
        lon1 = traf.lon[acid1]
        hdg1 = traf.hdg[acid1]
        
        # lat and lon and hdg of acid2
        lat2 = traf.lat[acid2]
        lon2 = traf.lon[acid2]
        hdg2 = traf.hdg[acid2]
        
        # call the qdrdist function to calculate distance
        qdr, distNM = tools.geo.qdrdist(lat1, lon1, lat2, lon2)
        
        # convert to meters
        dist = distNM * tools.aero.nm
        
        # calc difference in heading
        dhdg = hdg1 - hdg2
        
        return dist, qdr, dhdg
    
    @stack.command
    def echodistac(self, acid1: 'acid', acid2: 'acid'):
        
        #Calculate the distance and bearing 
        dist, qdr, dhdg = self.distAc(acid1, acid2)
        
        # round the output for the stack echo
        dist = round(dist,2)
        qdr = round(qdr,2)
        dhdg = round(dhdg,2)
        
        return True, f'Distance, bearing and heading from {traf.id[acid1]} to {traf.id[acid2]} is {dist} [m], {qdr} [deg] and {dhdg} [deg]'
    
    