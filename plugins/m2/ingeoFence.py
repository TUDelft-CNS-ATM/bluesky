""" BlueSky plugin template. The text you put here will be visible
    in BlueSky as the description of your plugin. """
from random import randint
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from bluesky import core, stack, traf, tools, settings

geofences = tools.areafilter.basic_shapes
geofence_names = geofences.keys()


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

    @core.timed_function(name='ingeofence', dt=settings.asas_dt)
    def update(self):
        #check for each aircraft if it interferes with one of the geofences
        for i in traf.id:
            idx=traf.id2idx(i)
            val = self.checker(acid=idx)
            self.ingeofence[idx] = val

    def checker(self, acid: 'acid'):
        multiGeofence = []
        # check for each geofence if the aircrafts intent intersects with the geofence
        # TODO Check if we can only run below function if a new geofence gets created... sort of like the super.create
        for j in geofence_names:

            # restructure the coordinates of the BS Poly shape to cast it into a shapely Polygon
            coord_list = list(zip(geofences[j].coordinates[1::2],geofences[j].coordinates[0::2]))

            #construct shapely Polygon object and add it to the multipolygon list
            shapely_geofence = Polygon(coord_list)
            multiGeofence.append(shapely_geofence)

        # get the aircraft intent to check against current geofence
        # at startup aircraft dont have intent yet, if so, return False
        acintent = traf.intent[acid]

        #construct the multipolygon object from all the polygons
        #this way you only have to check each aircraft against one shapely object instead of when each geofence in its own.
        multiGeofence = MultiPolygon(multiGeofence)

        #check for intersect between acintent and multipolygon
        val = acintent[0].intersects(multiGeofence)

        return val

    @stack.command
    def echoacgeofence(self, acid: 'acid'):
        ''' Print the if an acid is in conflict with a geofence '''
        geofence = self.getacgeofence(acid)
        return True, f'{traf.id[acid]} geofence conflict {geofence}.'

    def getacgeofence(self, acid: 'acid'):
        ''' return the bool value in ingeofence of a specific acid '''
        val = self.ingeofence[acid]
        return val