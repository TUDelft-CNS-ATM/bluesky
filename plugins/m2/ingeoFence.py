""" BlueSky plugin template. The text you put here will be visible
    in BlueSky as the description of your plugin. """
from random import randint
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, LineString
from bluesky import core, stack, traf, tools


geofence_names = tools.areafilter.basic_shapes.keys()
geofences = tools.areafilter.basic_shapes
acintent = traf.intent


def init_plugin():
    ingeofence = ingeoFence()
    config = {
        # The name of your plugin
        'plugin_name':     'ingeofence',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',
        }

    # init_plugin() should always return a configuration dict.
    return config

class ingeoFence(core.Entity):
    ''' Example new entity object for BlueSky. '''
    def __init__(self):
        super().__init__()
        with self.settrafarrays():
            self.ingeofence = np.array([],dtype=bool)

    @core.timed_function(name='ingeofence', dt=3)
    def update(self):
        for i in traf.id:
            idx=traf.id2idx(i)
            val = self.checker(acid=idx)
            self.ingeofence[idx] = val

    def checker(self, acid: 'acid'):
        #list that shows true or false per geofence for this specific aircraft
        infence=[]

        # check for each geofence if the aircrafts intent intersects with the geofence
        for j in geofence_names:

            # restructure the coordinates of the BS Poly shape to cast it into a shapely Polygon
            coord_list = list(zip(geofences[j].coordinates[1::2],geofences[j].coordinates[0::2]))

            #construct shapely Polygon object
            shapely_geofence = Polygon(coord_list)

            # get the aircraft intent to check against current geofence
            acintent = traf.intent[acid]
            val = acintent[0].intersects(shapely_geofence)
            infence.append(val)

        #check if the aircraft intersects with any of the geofences, if yes return True
        if any(infence):
            return True
        #if not return False, also meant to reset old Trues back to False when neccessary
        else:
            return False