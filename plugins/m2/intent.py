""" The intent plugin computes the intent of each aircraft with the CD lookahead
in both horizontal and vertical directions and stores this in the traffic object.
Created by: Emmanuel and Andrei
Date: 27 July 2021
"""
from shapely.geometry import LineString

# Import the global bluesky objects. Uncomment the ones you need
from bluesky import core, traf, settings, stack#, navdb, sim, scr, tools
from bluesky.tools import geo
from bluesky.tools.aero import nm#, ft

### Initialization function of your plugin. Do not change the name of this
### function, as it is the way BlueSky recognises this file as a plugin.
def init_plugin():
    ''' Plugin initialisation function. '''
    # Instantiate our example entity
    Intent = intent()

    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'intent',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',
        }

    # init_plugin() should always return a configuration dict.
    return config


class intent(core.Entity):
    ''' Example new entity object for BlueSky. '''
    def __init__(self):
        super().__init__()
        
        # add acintent as a new variable per aircraft
        with self.settrafarrays():
            self.acintent = []
        
        # Flag for using intent filter
        traf.swintent = True 
        
        # add intent to traffic make it available in the rest of bluesky and other plugins
        traf.intent = self.acintent
        

    # Functions that need to be called periodically can be indicated to BlueSky
    # with the timed_function decorator
    @core.timed_function(name='example', dt=settings.asas_dt/2)
    def update(self):
        
        # calculate intent if the switch is on
        if traf.swintent:
            self.calc_intent() 
        # self.calc_intent() 
        
        # update the traffic variable
        traf.intent = self.acintent


    def calc_intent(self):
        ''''This function computes the intent of each aircraft up to the CD look-ahead time '''
        
        ownship = traf
        ntraf = ownship.ntraf
        
        for idx in range(ntraf):
            
            #------------ Vertical----------------
            # If there is a vertical maneuver going on, target altitude is intent.
            # Otherwise, intent is simply current altitude.
            # iwpid = traf.ap.route[idx].findact(idx)
            # if iwpid > -1:
            #     # There is a maneuver going on
            #     intentAlt = ownship.selalt[idx]
            # else:
            #     # No maneuver going on
            #     intentAlt = ownship.alt[idx]
            intentAlt = ownship.selalt[idx]
                
            #------------Horizontal-----------------
            # First, get route
            ac_route = ownship.ap.route[idx]
            # Current waypoint index
            wpt_i = ac_route.iactwp
            # Current aircraft data
            ac_lat = ownship.lat[idx]
            ac_lon = ownship.lon[idx]
            ac_tas = ownship.tas[idx]
            # Initialize stuff
            prev_lat = ac_lat
            prev_lon = ac_lon
            distance = 0 # Keep this in [m]
            # First point in intent line is the position of the aircraft itself
            linecoords = [(ac_lon, ac_lat)]
            # Target distance
            distance_max = ac_tas * traf.cd.dtlookahead[idx]
            
            
            while True:
                # Stop if there are no waypoints, just create a line with current position and projected
                # Also stop if we ran out of waypoints.
                if wpt_i < 0 or wpt_i == len(ac_route.wpname):
                    # Just make a line between current position and the one within dtlookahead
                    end_lat, end_lon = geo.kwikpos(prev_lat, prev_lon, ownship.trk[idx], distance_max / nm)
                    intentLine = LineString([(prev_lon, prev_lat), (end_lon, end_lat)])
                    self.acintent[idx] = (intentLine, intentAlt)
                    break

                # Get waypoint data
                wpt_lat = ac_route.wplat[wpt_i]
                wpt_lon = ac_route.wplon[wpt_i]
                
                qdr_to_next, distance_to_next  = geo.kwikqdrdist(prev_lat, prev_lon, wpt_lat, wpt_lon)
                distance_to_next = distance_to_next * nm # Now in meters
                
                if (distance + distance_to_next) < distance_max:
                    # Next waypoint is closer than distance_max
                    distance = distance + distance_to_next
                    # Simply take the waypoint as the next point in the line
                    linecoords.append((wpt_lon, wpt_lat))
                    prev_lat = wpt_lat
                    prev_lon = wpt_lon
                    # Go to next waypoint
                    wpt_i += 1
                    
                else:
                    # We have reached the max distance, but the next waypoint is further away
                    # Create a point a certain distance from the previous point and take that
                    # as the next intent waypoint. 
                    # Distance to new temp waypoint
                    travel_dist = (distance_max - distance) / nm # Convert to NM
                    # Calculate new waypoint
                    end_lat, end_lon = geo.kwikpos(prev_lat, prev_lon, qdr_to_next, travel_dist)
                    # Do the same thing as before
                    linecoords.append((end_lon, end_lat))
                    intentLine = LineString(linecoords)
                    # Append intentLine to overall intent
                    self.acintent[idx] = (intentLine, intentAlt)
                    # Stop the while loop, go to next aircraft
                    break
                
    
    @stack.command
    def intentactive(self, active: 'onoff'):
        '''Set the intent filter on/off'''
        
        traf.swintent = active
        
        return True, 'Intent Filter is now ON' if active else 'Intent Filter is now OFF'
