import numpy as np
from ..tools.aero import ft

class Area:
    def __init__(self,traf):
        self.traf   = traf
        self.active = False

        # Traffic area: delete traffic when it leaves this area (so not when outside)
        self.lat0   = 0.0     # [deg] lower latitude defining area
        self.lat1   = 0.0     # [deg] upper latitude defining area
        self.lon0   = 0.0     # [deg] lower longitude defining area
        self.lon1   = 0.0     # [deg] upper longitude defining area
        self.floor  = -999.0  # [m] Delete when descending through this h
        self.dt     = 5.0     # [s] frequency of area check (simtime)
        self.t0     = -100.   # last time checked
        self.radius = 100.0   # [NM] radius of experiment area if it is a circle    

        self.shape = "" # "Square" for square, "Circle" for circle
        
        # Taxi switch
        self.swtaxi = False  # Default OFF: delete traffic below 1500 ft
        
        # Boolean array whether aircraft are in circle or not
        self.inside = np.array([],dtype = np.bool) # In test area or not

    def create(self):
        self.inside = np.append(self.inside,False)

    def delete(self,idx):
        self.inside = np.delete(self.inside,idx)

    def check(self,t):
        if not (self.active and t-self.t0 > self.dt):
            return
        self.t0 = t
 
        # Check all aircraft
        i = 0
        while (i < self.traf.ntraf):
            # Current status
            if self.shape == "Square":
                inside = self.lat0 <= self.traf.lat[i] <= self.lat1 and \
                         self.lon0 <= self.traf.lon[i] <= self.lon1 and \
                         self.traf.alt[i] >= self.floor and \
                         (self.traf.alt[i] >= 0.5*ft or self.swtaxi)

            elif self.shape == "Circle":

                # delete aircraft if it is too far from the center of the circular area, or if has decended below the minimum altitude
                distance = geo.kwikdist(self.lat0, self.lon0, self.traf.lat[i], self.traf.lon[i])  # [NM]
                inside = distance < self.radius and self.traf.alt[i] >= self.floor

            # Compare with previous: when leaving area: delete command
            if self.inside[i] and not inside:
                self.traf.delete(self.traf.id[i])

            else:
                # Update area status
                self.inside[i] = inside
                i = i + 1

    def SetArea(self, scr, metric, *args):
        if args[0] == 'OFF':
            self.active = False
            self.shape   = ""
            scr.objappend('BOX', "AREA", None)  # delete square areas
            scr.objappend('CIRCLE', "AREA", None)  # delete circle areas
            return True

        if type(args[0]) == float and len(args) >= 4:
            # This is a square area
            self.lat0 = min(args[0], args[2])
            self.lat1 = max(args[0], args[2])
            self.lon0 = min(args[1], args[3])
            self.lon1 = max(args[1], args[3])

            if len(args) == 5:
                self.floor = args[4] * ft
            else:
                self.floor = -9999999.

            self.shape = "Square"
            self.active = True
            scr.objappend('BOX', "AREA", [args[0], args[1], args[2], args[3]])

            # Avoid mass delete due to redefinition of area
            self.inside.fill(False)
            return True

        elif args[0] == "FIR" and len(args) <= 3:
            for i in range(0, len(self.traf.navdb.fir)):
                if args[1] == self.traf.navdb.fir[i][0]:
                    break
            if args[1] != self.traf.navdb.fir[i][0]:
                return False, "Unknown FIR, try again"

            metric.fir_number        = i
            metric.fir_circle_point  = metric.metric_Area.FIR_circle(self.traf.navdb, metric.fir_number)
            metric.fir_circle_radius = float(args[1])

            if len(args) == 3:
                self.floor = args[2] * ft
            else:
                self.floor = -9999999.

            self.shape   = "Circle"
            self.active = True
            self.inside.fill(False)
            scr.objappend('CIRCLE', "AREA", [metric.fir_circle_point[0] , metric.fir_circle_point[1], metric.fir_circle_radius])
            return True

        elif args[0] == "CIRCLE" and len(args) in [4, 5]:
            # draw circular experiment area
            self.lat0 = args[1]    # Latitude of circle center [deg]
            self.lon0 = args[2]    # Longitude of circle center [deg]
            self.radius = args[3]  # Radius of circle Center [NM]

            # Deleting traffic flying out of experiment area
            self.shape = "Circle"
            self.active = True

            if len(args) == 5:
                self.floor = args[4] * ft  # [m]
            else:
                self.floor = -9999999.  # [m]

            # draw the circular experiment area on the radar gui
            scr.objappend('CIRCLE', "AREA", [self.lat0, self.lon0, self.radius])

            # Avoid mass delete due to redefinition of area
            self.inside.fill(False)

            return True

        return False

    def setTaxi(self, flag):
        """ Set taxi delete flag: OFF auto deletes traffic below 1500 ft """
        self.swtaxi = flag