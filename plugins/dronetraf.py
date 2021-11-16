""" Creates random traffic of drones within a set area at regular intervals. """
from random import randint, choice, uniform
import numpy as np
# Import the global bluesky objects. Uncomment the ones you need
from bluesky import core, stack, traf  #, settings, navdb, sim, scr, tools

CHANCE_OF_TRAFFIC = 30

# List of IDs in use
drones_active = list()

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
    ''' Generates new drones and maintains a list of active drones. '''

    def __init__(self):
        super().__init__()

    # Called automatically when a done is deleted
    def delete(self, idx):
        super().delete(idx)
        drones_active.remove(traf.id[idx[0]])

    # Called every 10 simulation steps(seconds)
    @core.timed_function(name='drone_traffic', dt=10)
    def update(self):
        new_traffic = randint(1, 100)
        if new_traffic <= CHANCE_OF_TRAFFIC:
            self.create_drone()

    def create_drone(self):
        acid = self.assign_id()

        # Origin chosen at random from predefined points outside the area of interest
        origins = ["52.44853, 9.62503", "52.41137, 9.65701", "52.40540, 9.69458", "52.41673, 9.73632", "52.44602, 9.75462"]
        origin = {}
        origin["lat"], origin["lon"] = choice(origins).split(", ")
        # Destination point chosen at random fom within the area of interest
        dst = {}
        dst["lat"] = str(round(uniform(52.42971, 52.44556), 5))
        dst["lon"] = str(round(uniform(9.67770, 9.71830), 5))

        # Orienting the drone towards the destination, as too great a turn breaks the autopilot
        if float(origin["lat"]) < float(dst["lat"]):
            hdg = "0"
        else:
            hdg = "180"
        alt = str(randint(100, 350))
        spd = str(randint(15, 34))
        
        # Sending the commands to the simulator
        stack.stack("CRE " + acid + ", M600, " + origin["lat"] + ", " + origin["lon"] + ", " + hdg + ", " + alt + ", " + spd)
        stack.stack("DEST " + acid + ", " + dst["lat"] + ", " + dst["lon"])
        stack.stack(acid + " ATALT 0 DEL " + acid)

    # Assigns a new ID not already in use
    def assign_id(self):
        approved = False

        while not approved:
            newid = "D" + str(randint(1, 999))
            if newid not in drones_active: approved = True

        drones_active.append(newid)
        return newid
