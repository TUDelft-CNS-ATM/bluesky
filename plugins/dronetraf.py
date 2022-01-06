""" Creates random traffic of drones within a set area at regular intervals. """
from random import randint, uniform
import numpy as np
# Import the global bluesky objects. Uncomment the ones you need
from bluesky import core, stack, traf  #, settings, navdb, sim, scr, tools
from bluesky.tools import areafilter

CHANCE_OF_TRAFFIC = 30

### Initialisation function of your plugin. Do not change the name of this
### function, as it is the way BlueSky recognises this file as a plugin.
def init_plugin():
    ''' Plugin initialisation function. '''
    # Instantiate the Dronetraf entity
    dronetraf = Dronetraf()

    # Configuration parameters
    config = {
        'plugin_name':     'DRONETRAF',
        'plugin_type':     'sim',
        }
    
    stackfunctions = {
        'DRONETRAF': [
            'DRONETRAF cmd, [area]',
            'txt,[txt]',
            dronetraf.dronetraf,
            'Define the area of operation as any of the shapes available in the simulator.'
        ]
    }

    # init_plugin() should always return a configuration dict.
    return config, stackfunctions

class Dronetraf(core.Entity):
    ''' Generates new drones within a given area and maintains a list of active drones. '''

    def __init__(self):
        super().__init__()

        # List of IDs in use
        self.drones_active = list()
        # Area of operation
        self.area = ''
        self.area_bounding_box = []

    # Called automatically when a drone is deleted
    def delete(self, idx):
        ''' Removes the deleted drone's ID from the list. '''
        super().delete(idx)
        if traf.id[idx[0]] in self.drones_active:
            self.drones_active.remove(traf.id[idx[0]])

    # Called every 5 simulation steps(seconds)
    @core.timed_function(name='drone_traffic', dt=5)
    def update(self):
        # Only generate drones if the area is set
        if len(self.area) > 0:
            new_traffic = randint(1, 100)
            if new_traffic <= CHANCE_OF_TRAFFIC:
                self.create_drone()

    def create_drone(self):
        ''' Creates a drone within the the stated area and assigns it a flight plan. '''
        acid = self.assign_id()

        # Origin and destination chosen at random from within the area of operation
        origin = self.assign_wpt(self.area_bounding_box[0], self.area_bounding_box[1])
        dst = self.assign_wpt(self.area_bounding_box[0], self.area_bounding_box[1])

        # Orienting the drone towards the destination, as too great a turn breaks the autopilot
        if float(origin["lat"]) < float(dst["lat"]):
            hdg = "0"
        else:
            hdg = "180"
        alt = str(randint(100, 350))
        spd = str(randint(15, 34))
        
        # Sending the commands to the stack
        stack.stack(f'CRE {acid} M600 {origin["lat"]} {origin["lon"]} {hdg} {alt} {spd}')
        stack.stack(f'ADDWPT {acid} {dst["lat"]} {dst["lon"]}')
        stack.stack(f'{acid} ATALT 0 DEL {acid}')
        stack.stack(f'VS {acid} 900')
        stack.stack(f'VNAV {acid} ON')
        stack.stack(f'{acid} ATDIST {dst["lat"]} {dst["lon"]} 0.03 SPD {acid} 10')
        stack.stack(f'{acid} ATDIST {dst["lat"]} {dst["lon"]} 0.001 SPD {acid} 0')
        stack.stack(f'{acid} ATSPD 0 ALT {acid} 0')

    def assign_id(self):
        ''' Assigns a new ID not already in use. '''
        approved = False

        while not approved:
            newid = "D" + str(randint(1, 999))
            if newid not in self.drones_active: approved = True

        self.drones_active.append(newid)
        return newid

    def assign_wpt(self, coord1, coord2):
        ''' Returns the coordinate of a random point within the set area, by utilising its bounding box. '''
        wpt = {}
        inside = False

        while not inside:
            lat = round(uniform(coord1["lat"], coord2["lat"]), 5)
            lon = round(uniform(coord1["lon"], coord2["lon"]), 5)
            inside = areafilter.checkInside(self.area, lat, lon, 0)
        
        wpt["lat"] = str(lat)
        wpt["lon"] = str(lon)

        return wpt
    
    def set_area(self, area):
        ''' Sets the area of operation, together with its bounding box. '''
        self.area = area
        coords = areafilter.basic_shapes[area].coordinates
        lat0 = min(coords[::2])
        lon0 = min(coords[1::2])
        lat1 = max(coords[::2])
        lon1 = max(coords[1::2])

        self.area_bounding_box = [
            {"lat": lat0, "lon": lon0},
            {"lat": lat1, "lon": lon1}]

    def dronetraf(self, cmd, area=''):
        ''' The commands available for the plugin.
            AREA: Takes the name of a defined shape and sets it as the area. '''
        if cmd == 'AREA':
            if area == self.area:
                if area == '':
                    return False, f'Area name must be specified.'
                return True, f'Area {area} is already set.'
            elif areafilter.hasArea(area):
                self.set_area(area)
                return True, f'{area} has been set as the new area for drone traffic.'
            else:
                return False, f'No area found with name "{area}", create it first with one of the shape commands.'
        else:
            return False, f'Available commands are: AREA'
