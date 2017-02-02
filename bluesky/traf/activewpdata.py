import numpy as np
from ..tools.dynamicarrays import DynamicArrays, RegisterElementParameters
from ..tools.aero import nm, g0
from ..tools.misc import degto180


class ActiveWaypoint(DynamicArrays):
    def __init__(self, traf):
        self.traf = traf

        with RegisterElementParameters(self):
            self.lat      = np.array([])  # Active WP latitude
            self.lon      = np.array([])  # Active WP longitude
            self.alt      = np.array([])  # Active WP altitude to arrive at
            self.spd      = np.array([])  # Active WP speed
            self.vs       = np.array([])  # Active vertical speed to use
            self.turndist = np.array([])  # Distance when to turn to next waypoint
            self.flyby    = np.array([])  # Distance when to turn to next waypoint
            self.next_qdr = np.array([])  # bearing next leg

    def create(self):
        super(ActiveWaypoint, self).create()
        # LNAV route navigation
        self.lat[-1]       = 89.99  # Active WP latitude
        self.spd[-1]       = -999.   # Active WP speed
        self.turndist[-1]  = 1.0   # Distance to active waypoint where to turn
        self.flyby[-1]     = 1.0   # Flyby/fly-over switch
        self.next_qdr[-1]  = -999.0    # bearing next leg

    def Reached(self, qdr, dist):
        # Calculate distance before waypoint where to start the turn
        # Turn radius:      R = V2 tan phi / g
        # Distance to turn: wpturn = R * tan (1/2 delhdg) but max 4 times radius
        # using default bank angle per flight phase
        turnrad = self.traf.tas * self.traf.tas / np.maximum(self.traf.eps, np.tan(self.traf.bank) * g0 * nm)  # [nm]
        next_qdr = np.where(self.next_qdr < -900., qdr, self.next_qdr)
     
        # Avoid circling
        away = np.abs(degto180(self.traf.trk - next_qdr)+180.)>90.
        incircle = dist<turnrad*1.01
        circling = away*incircle


        # distance to turn initialisation point
        self.turndist = np.minimum(100., np.abs(turnrad *
            np.tan(np.radians(0.5 * np.abs(degto180(qdr - next_qdr))))))

        # Check whether shift based dist [nm] is required, set closer than WP turn distanc
        return np.where(self.traf.swlnav * ((dist < self.turndist)+circling))[0]
