from bluesky import settings, stack
from bluesky.tools import aero, areafilter, geo

settings.set_variable_defaults(geofence_dtlookahead=30)

def init_plugin():
    # Configuration parameters
    config = {
        'plugin_name': 'GEOFENCE',
        'plugin_type': 'sim',
        'reset': Geofence.reset
    }
    return config

@stack.command()
def geofence(name: 'txt', top: float, bottom: float, *coordinates: float):
    ''' Create a new geofence from the stack. 
        
        Arguments:
        - name: The name of the new geofence
        - top: The top of the geofence in feet.
        - bottom: The bottom of the geofence in feet.
        - coordinates: three or more lat/lon coordinates in degrees.
    '''
    Geofence.geofences[name] = Geofence(name, top, bottom, coordinates)
    return True, f'Created geofence {name}'

class Geofence(areafilter.Poly):
    ''' BlueSky Geofence class.
    
        This class subclasses Shape, and adds Geofence-specific data and methods.
    '''
    # Keep a dict of geofences
    geofences = dict()

    def __init__(self, name, coordinates, top, bottom):
        super().__init__(name, coordinates, top=top, bottom=bottom)
        self.active = True

    def intersects(self, line):
        ''' Check whether given line intersects with this geofence poly. '''
        pass

    @classmethod
    def reset(cls):
        ''' Reset geofence database when simulation is reset. '''
        cls.geofences.clear()

    @classmethod
    def detect_all(cls, traf, dtlookahead=None):
        if dtlookahead is None:
            dtlookahead = settings.geofence_dtlookahead
        
        # Linearly extrapolate current state to prefict future position
        pred_lat, pred_lon = geo.kwikpos(traf.lat, traf.lon, traf.hdg, traf.gs / aero.nm)
        hits_per_ac = []
        for idx, line in zip(traf.lat, traf.lon, pred_lat, pred_lon):
            hits = []
            # First a course detection based on geofence bounding boxes
            potential_hits = areafilter.get_intersecting(*line)
            # Then a fine-grained intersection detection
            for geofence in potential_hits:
                if geofence.intersects(line):
                    hits.append(geofence)
            hits_per_ac.append(hits)

        return hits_per_ac
