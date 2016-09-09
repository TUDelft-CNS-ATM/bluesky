import numpy as np
from ..tools.dynamicarrays import DynamicArrays, RegisterElementParameters
from ..tools.aero import nm, g0
from ..tools.misc import degto180

class ActiveWaypoint(DynamicArrays):
    def __init__(self,fms):
        self.fms = fms
        self.traf = fms.traf
        
        with RegisterElementParameters(self):
            self.lat  = np.array([])  # Active WP latitude
            self.lon  = np.array([])  # Active WP longitude
            self.alt  = np.array([])  # Active WP altitude to arrive at
            self.spd  = np.array([])  # Active WP speed
            self.vs   = np.array([])  # Active vertical speed to use
            self.turn = np.array([])  # Distance when to turn to next waypoint
            self.flyby = np.array([])  # Distance when to turn to next waypoint
            self.next_qdr  = np.array([])  # bearing next leg

    def create(self):
        self.CreateElement()
        # LNAV route navigation
        self.lat[-1]       = 89.99  # Active WP latitude
        self.spd[-1]       = -999.   # Active WP speed
        self.turn[-1]      = 1.0   # Distance to active waypoint where to turn
        self.flyby[-1]     = 1.0   # Flyby/fly-over switch
        self.next_qdr[-1]  = -999.0    # bearing next leg

    def delete(self,idx):
        self.DeleteElement(idx)

    def Reached(self, qdr, dist):
        # Calculate distance before waypoint where to start the turn
        # Turn radius:      R = V2 tan phi / g
        # Distance to turn: wpturn = R * tan (1/2 delhdg) but max 4 times radius
        # using default bank angle per flight phase
        turnrad = self.traf.tas*self.traf.tas / np.maximum(self.traf.eps,np.tan(self.traf.bank)*g0*nm) # [nm]
        next_qdr = np.where(self.next_qdr < -900., qdr, self.next_qdr)

        # distance to turn initialisation point
        self.turn = np.maximum(100, np.abs(turnrad*np.tan(np.radians(0.5*degto180(np.abs(qdr -    \
             next_qdr))))))

        # Check whether shift based dist [nm] is required, set closer than WP turn distanc
        return np.where(self.fms.lnav * (dist < self.turn))[0]