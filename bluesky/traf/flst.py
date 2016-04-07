import numpy as np
from ..tools.aero import ft, kts, g0, qdrdist, nm, cas2tas, mach2tas
from ..tools.aero_np import vtas2eas
from ..tools.misc import degto180


class FLST():
    """ 
    Flight Statistics class definition   : flight statistics for all aircraft
    
    Created by  : Martijn Tra
    """

    def __init__(self,traf):
        # Add pointer to self traf object
        self.traf  = traf

        # Reset the recording of the flight statistics
        self.reset()

        return

    def reset(self):
        # Flight Statistics data
        self.distance_2D    = np.array([]) # 2-dimensional flight distance [m]
        self.distance_3D    = np.array([]) # 3-dimensional flight distance [m]
        self.flightime      = np.array([]) # Total flight time [sec]
        self.work           = np.array([]) # Work done [GJ]

    def update(self,simdt):
        # ------------- FLIGHT STATISTICS ----------
        # Flight Statistics data
        self.distance_2D    = self.distance_2D + simdt * self.traf.gs # [m]
        self.distance_3D    = self.distance_3D + simdt * np.sqrt(self.traf.gs**2 + self.traf.vs**2) # [m]
        self.flightime      = self.flightime + simdt # [sec]
        self.work           = self.work + (self.traf.perf.Thr * simdt * vtas2eas(self.traf.tas,self.traf.alt))/10**9 # [GJ]: Giga Joule -> work = thrust * ds, with ds = simdt * tas
