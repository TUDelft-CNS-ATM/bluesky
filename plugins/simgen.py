from bluesky.core import Entity, timed_function
from bluesky import core, stack, traf  #, settings, navdb, sim, scr, tools
from random import randint 
import numpy as np
from bluesky.tools.aero import fpm, kts, ft, g0 #, Rearth, nm, tas2cas,\
                        # vatmos,  vtas2cas, vtas2mach, 

LGW_LAT = 51.1537
LGW_LON = 0.1821

SID = [(i + LGW_LAT, i + LGW_LON) for i in range(0, 10, 2)]
STAR = [(LGW_LAT - i, LGW_LON - i) for i in range(10, -1, -2)]
WPT_NAMES = ['WPT' + str(i) for i in range(0, len(SID))]
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
        'plugin_type':     'sim', 

        #'update' : update

        }
    return config 


def get_SID():
    return SID

def get_STAR():
    return STAR

class Simgen(Entity):

    nextwpt = []
    SID = get_SID()
    STAR = get_STAR()
    sidstar = []

    def __init__(self):
        super.__init__(self)

        #with self.settrafarrays():
            #self.sidstar = np.array([])
            #self.nextwpt = np.array([])
        
    def create(self, n = 1):
        ''' This function gets called automatically when new aircraft are created. '''
        super().create(n)
        
        
        #Aircraft ids and types
        idtmp = chr(randint(65, 90)) + chr(randint(65, 90)) + '{:>05}'
        traf.id[-n:] = [idtmp.format(i) for i in range(n)]
        traf.type[-n:] = [ACTYPES[randint(0, len(ACTYPES))] for _ in range(n)]

        #Nature of the flight: either departing or arriving
        #0 for departing, 1 for arrival
        Simgen.sidstar.extend([randint(0, 1) for _ in range(n)])
        
        #POSITION 
        
        for i in range(n):
            if Simgen.sidstar[i] == 0: #departing
                traf.lat[i], traf.lon[i] = Simgen.SID[0][0] - SPAWN_RADIUS(), \
                            Simgen.SID[0][1] - SPAWN_RADIUS()
            else: #arriving
                traf.lat[i], traf.lon[i] = Simgen.STAR()[0][0] - SPAWN_RADIUS(), \
                            Simgen.STAR()[0][1] - SPAWN_RADIUS()
        
        Simgen.nextwpt.extend([1 for _ in range(n)]) #index of the next SIDSTART waypoint
        traf.alt[-n:] = [randint(WPT_MAX, WPT_MIN) * ft for _ in range(n)]
        
        #Velocity and headings
        #self.trk[-n:] = [np.random.randint(10, 90) for _ in range(n)] # random headings at the moment
        traf.tas[-n:] = [np.random.randint(250, 450, n) * kts for _ in range(n)]
        
        #adding waypoints

        for i in range(n):
            # if self.sidstar[i] == 0: #departing
            #     self.actwp.lat[i], self.actwp.lon[i] = self.SID[1]
            # else:
            #     self.actwp.lat[i], self.actwp.lon[i] = self.STAR[1]
            acid = traf.id[i]
            
            stack.stack(f'ECHO Adding initial waypoint {Simgen.SID[0] if Simgen.sidstar[i] == 0 else Simgen.STAR[0]}' +
                        'to {acid}')

            stack.stack(f'ADDWPT {acid}, {WPT_NAMES[0]}, {Simgen.SID[0][0] if Simgen.sidstar[i] == 0 else Simgen.STAR[0][0]},' + 
                        f'{Simgen.SID[0][1] if Simgen.sidstar[i] == 1 else Simgen.STAR[0][1]}, FLYBY, {traf.alt[i] + randint(0, 100)}, ' +
                        f'{traf.tas[i] + randint(-100, 100)}')

    @classmethod
    @timed_function(dt=5)
    def update(cls):
        ''' Periodic update function for our example entity. '''
        
        n = randint(1, 4)
        cls.nextwpt.extend([1 for _ in range(n)])
        stack.stack(f"MCRE {n}")
        Simgen.sidstar.extend([randint(0, 1) for _ in range(n)])
        #self.create(n = randint(1, 4))
        for i in range(traf.ntraf):
            #get current waypoint position and next waypoint index
            #currwpt = self.actwp.lat[i], self.actwp.lon[i]
            nextwpt = cls.nextwpt[i]
            
            if (nextwpt):

                acid = traf.id[i]

                stack.stack(f'ECHO Adding next waypoint {cls.SID[nextwpt] if cls.sidstar[i] == 0 else cls.STAR[nextwpt]} called' +
                            f' {WPT_NAMES[nextwpt]} to {acid}')
                
                stack.stack(f'{acid} AFTER {WPT_NAMES[nextwpt - 1]} ADDWPT {WPT_NAMES[nextwpt]},' + 
                            f'{cls.SID[nextwpt][0] if cls.sidstar[i] == 0 else cls.STAR[nextwpt][0]}, ' + 
                            f'{cls.SID[nextwpt][1] if cls.sidstar[i] == 0 else cls.STAR[nextwpt][1]},' + 
                            f'{traf.tas[i] + randint(-100, 100)}, {traf.alt[i] + randint(-100, 100)}')

            else:

                stack.stack(f'ECHO Adding initial waypoint WPT0 to {acid}')
                stack.stack(f'ADDPT {acid}, WPT0, ' +
                            f'{cls.SID[nextwpt][0] if cls.sidstar[i] == 0 else cls.STAR[nextwpt][0]}, ' + 
                            f'{cls.SID[nextwpt][1] if cls.sidstar[i] == 0 else cls.STAR[nextwpt][1]},' + 
                            f'{traf.tas[i] + randint(-100, 100)}, {traf.alt[i] + randint(-100, 100)}')

            cls.nextwpt[i] = cls.nextwpt[i] + 1
                

