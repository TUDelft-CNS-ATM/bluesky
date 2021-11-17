""" Creates random traffic of drones within a set area at regular intervals. """
from random import randint, choice, uniform
import numpy as np
# Import the global bluesky objects. Uncomment the ones you need
from bluesky import core, stack, traf  #, settings, navdb, sim, scr, tools

CHANCE_OF_TRAFFIC = 30

### Initialization function of your plugin. Do not change the name of this
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

    # init_plugin() should always return a configuration dict.
    return config

class Dronetraf(core.Entity):
    ''' Generates new drones within a given area and maintains a list of active drones. '''

    def __init__(self):
        super().__init__()

        # List of IDs in use
        self.drones_active = list()
        # Area of interest
        self.area_coords = [
            {"lat": 52.42738, "lon": 9.65979},
            {"lat": 52.45014, "lon": 9.72837}]

    # Called automatically when a drone is deleted
    def delete(self, idx):
        super().delete(idx)
        self.drones_active.remove(traf.id[idx[0]])

    # Called every 5 simulation steps(seconds)
    @core.timed_function(name='drone_traffic', dt=5)
    def update(self):
        new_traffic = randint(1, 100)
        if new_traffic <= CHANCE_OF_TRAFFIC:
            self.create_drone()

    def create_drone(self):
        acid = self.assign_id()

        # Origin and destination chosen at random from within the area of interest
        origin = self.assign_wpt(self.area_coords[0], self.area_coords[1])
        dst = self.assign_wpt(self.area_coords[0], self.area_coords[1])

        # Orienting the drone towards the destination, as too great a turn breaks the autopilot
        if float(origin["lat"]) < float(dst["lat"]):
            hdg = "0"
        else:
            hdg = "180"
        alt = str(randint(100, 350))
        spd = str(randint(15, 34))
        
        # Sending the commands to the simulator
        stack.stack("CRE " + acid + ", M600, " + origin["lat"] + ", " + origin["lon"] + ", " + hdg + ", " + alt + ", " + spd)
        stack.stack("ADDWPT " + acid + ", " + dst["lat"] + ", " + dst["lon"])
        stack.stack(acid + " ATALT 0, DEL " + acid)
        stack.stack("VS " + acid + ", 900")
        stack.stack("VNAV " + acid + ", ON")
        stack.stack(acid + " AT " + acid + "001 DO SPD " + acid + " 0")
        stack.stack(acid + " AT " + acid + "001 DO ALT " + acid + " 0")

    # Assigns a new ID not already in use
    def assign_id(self):
        approved = False

        while not approved:
            newid = "D" + str(randint(1, 999))
            if newid not in self.drones_active: approved = True

        self.drones_active.append(newid)
        return newid
    
    # Takes 2 coordinates and returns a point from within the rectangle they form
    def assign_wpt(self, coord1, coord2):
        wpt = {}
        wpt["lat"] = str(round(uniform(coord1["lat"], coord2["lat"]), 5))
        wpt["lon"] = str(round(uniform(coord1["lon"], coord2["lon"]), 5))

        return wpt
