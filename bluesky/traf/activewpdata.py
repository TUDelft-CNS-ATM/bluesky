import numpy as np
import bluesky as bs
from bluesky.tools.trafficarrays import TrafficArrays, RegisterElementParameters
from bluesky.tools.aero import nm, g0
from bluesky.tools.misc import degto180


class ActiveWaypoint(TrafficArrays):
    def __init__(self):
        super(ActiveWaypoint, self).__init__()
        with RegisterElementParameters(self):
            self.lat      = np.array([])  # Active WP latitude
            self.lon      = np.array([])  # Active WP longitude
            self.alt      = np.array([])  # Altitude to arrive at after distance xtoalt
            self.xtoalt   = np.array([])  # Distance to next altitude constraint
            self.spd      = np.array([])  # Active WP speed
            self.vs       = np.array([])  # Active vertical speed to use
            self.turndist = np.array([])  # Distance when to turn to next waypoint
            self.flyby    = np.array([])  # Distance when to turn to next waypoint
            self.next_qdr = np.array([])  # bearing next leg

    def create(self, n=1):
        super(ActiveWaypoint, self).create(n)
        # LNAV route navigation
        self.lat[-n:]       = 89.99  # Active WP latitude
        self.spd[-n:]       = -999.   # Active WP speed
        self.turndist[-n:]  = 1.0   # Distance to active waypoint where to turn
        self.flyby[-n:]     = 1.0   # Flyby/fly-over switch
        self.next_qdr[-n:]  = -999.0    # bearing next leg

    def Reached(self, qdr, dist, flyby):
        # Calculate distance before waypoint where to start the turn
        # Turn radius:      R = V2 tan phi / g
        # Distance to turn: wpturn = R * tan (1/2 delhdg) but max 4 times radius
        # using default bank angle per flight phase
        turnrad = bs.traf.tas * bs.traf.tas /     \
                      np.maximum(bs.traf.eps, np.tan(bs.traf.bank) * g0 * nm)  # [nm]

        next_qdr = np.where(self.next_qdr < -900., qdr, self.next_qdr)

        # Avoid circling
#        away = np.abs(degto180(bs.traf.trk - next_qdr)+180.)>90.
#        away     = np.abs(degto180(bs.traf.trk - next_qdr))>90.
        away     = np.abs(degto180(bs.traf.trk%360. - qdr%360.))>90.
        incircle = dist<turnrad*1.01
        circling = away*incircle


        # distance to turn initialisation point [nm]
        self.turndist = flyby*np.minimum(100., np.abs(turnrad *
            np.tan(np.radians(0.5 * np.abs(degto180(qdr%360. - next_qdr%360.))))))


        # Check whether shift based dist [nm] is required, set closer than WP turn distanc
        return np.where(bs.traf.swlnav * ((dist < self.turndist)+circling))[0]
