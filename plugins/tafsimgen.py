from bluesky import core, stack, traf  #, settings, navdb, sim, scr, tools
from random import random 
import numpy as np
from bluesky.tools.aero import fpm, kts, ft, g0 #, Rearth, nm, tas2cas,\
                        # vatmos,  vtas2cas, vtas2mach, 

LGW_LAT = 51.1537
LGW_LON = 0.1821

SID = [(i + LGW_LAT, i + LGW_LON) for i in range(0, 10, 2)]
STAR = [(LGW_LAT - i, LGW_LON - i) for i in range(10, -1, -2)]
WPT_NAMES = ['WPT' + i for i in range(len(SID))]
WPT_MAX = 4000
WPT_MIN = 3000
SPAWN_RADIUS = lambda : randint(0, 10)
ACTYPES = ['B744', 'B747', 'B73V']

def init_plugin():
    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'SIMGEN',

        # The type of this plugin.
        'plugin_type':     'sim'

        }

    return config 

class Simgen(Entity):

    def __init__(self):
        super.__init__()

        with self.settrafarrays():
            self.sidstar = np.array([])
            self.nextwpt = np.array([])
        
        self.SID = get_SID()
        self.STAR = get_STAR()
        
    def create(self, n = 1):
        ''' This function gets called automatically when new aircraft are created. '''
        super().create(n)
        
        #Aircraft ids and types
        idtmp = chr(randint(65, 90)) + chr(randint(65, 90)) + '{:>05}'
        self.id[-n:] = [idtmp.format(i) for i in range(n)]
        self.type[-n:] = [ACTYPES[randint(0, len(ACTYPES))] for _ in range(n)]

        #Nature of the flight: either departing or arriving
        #0 for departing, 1 for arrival
        self.sidstar[-n:] = [randint(0, 1) for _ in range(n)]
        
        #POSITION 
        
        for i in range(n):
            if self.sidstar[i] == 0: #departing
                self.lat[i], self.lon[i] = self.SID[0][0] - SPAWN_RADIUS(), \
                            self.SID[0][1] - SPAWN_RADIUS()
            else: #arriving
                self.lat[i], self.lon[i] = self.STAR()[0][0] - SPAWN_RADIUS(), \
                            self.STAR()[0][1] - SPAWN_RADIUS()
        
        self.nextwpt[-n:] = [1 for _ in range(n)] #index of the next SIDSTART waypoint
        self.alt[-n:] = [randint(WPT_MAX, WPT_MIN) * ft for _ in range(n)]
        
        #Velocity and headings
        #self.trk[-n:] = [np.random.randint(10, 90) for _ in range(n)] # random headings at the moment
        self.tas[-n:] = [np.random.randint(250, 450, n) * kts for _ in range(n)]
        
        #adding waypoints

        for i in range(n):
            # if self.sidstar[i] == 0: #departing
            #     self.actwp.lat[i], self.actwp.lon[i] = self.SID[1]
            # else:
            #     self.actwp.lat[i], self.actwp.lon[i] = self.STAR[1]
            acid = self.id[i]
            
            stack.stack(f'ECHO Adding initial waypoint {self.SID[0] if self.sidstar[i] == 0 else self.STAR[0]} \
                        to {acid}')

            stack.stack(f'ADDWPT {acid}, {WPT_NAMES[0]}, {self.SID[0][0] if self.sidstar[i] == 0 else self.STAR[0][0]}, \
                {self.SID[0][1] if self.sidstar[i] == 1 else self.STAR[0][1]}, FLYBY, {self.alt[i] + randint(0, 100)}, \
                        {self.tas[i] + randint(-100, 100)}')


    @core.timed_function(name='simgen', dt=5)
    def update(self):
        ''' Periodic update function for our example entity. '''

        #self.create(n = randint(1, 4))
        for i in range(traf.ntraf):
            #get current waypoint position and next waypoint index
            currwpt = self.actwp.lat[i], self.actwp.lon[i]
            nextwpt = self.nextwpt[i]
            

            acid = self.id[i]

            stack.stack(f'ECHO Adding next waypoint {self.SID[nextwpt] \
                    if self.sidstar[i] == 0 else self.STAR[nextwpt]} to {acid}')
            
            stack.stack(f'{acid} AFTER {WPT_NAMES[nextwpt - 1]} ADDWPT {WPT_NAMES[nextwpt]}, \ 
                        {self.SID[nextwpt][0] if self.sidstar[i] == 0 else self.STAR[nextwpt][0]}, \
                        {self.SID[nextwpt][1] if self.sidstar[i] == 0 else self.STAR[nextwpt][1]}, \
                        {self.tas[i] + randint(-100, 100)}, {self.alt[i] + randint(-100, 100)}')

            self.nextwpt[i] = self.nextwpt[i] + 1