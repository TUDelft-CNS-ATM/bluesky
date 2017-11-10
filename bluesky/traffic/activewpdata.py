import numpy as np
import bluesky as bs
from bluesky.tools.trafficarrays import TrafficArrays, RegisterElementParameters
from bluesky.tools.aero import nm, g0
from bluesky.tools.misc import degto180


class ActiveWaypoint(TrafficArrays):
    def __init__(self):
        super(ActiveWaypoint, self).__init__()
        with RegisterElementParameters(self):
            self.lat       = np.array([])  # Active WP latitude
            self.lon       = np.array([])  # Active WP longitude
            self.nextaltco = np.array([])  # Altitude to arrive at after distance xtoalt
            self.xtoalt    = np.array([])  # Distance to next altitude constraint
            self.spd       = np.array([])  # Active WP speed
            self.vs        = np.array([])  # Active vertical speed to use
            self.turndist  = np.array([])  # Distance when to turn to next waypoint
            self.flyby     = np.array([])  # Distance when to turn to next waypoint
            self.next_qdr  = np.array([])  # bearing next leg

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
                      (np.maximum(bs.traf.eps, np.tan(bs.traf.bank)) * g0)  \
                       # Turn radius in meters!

        next_qdr = np.where(self.next_qdr < -900., qdr, self.next_qdr)

        # Avoid circling by checking for flying away
        away     = np.abs(degto180(bs.traf.trk%360. - qdr%360.)) > 90. # difference large than 90

        # Ratio between distance close enough to switch to next wp when flying away
        # When within pro1 nm and flying away: switch also
        proxfact = 1.01 # Turnradius scales this contant , factor => [turnrad]
        incircle = dist<turnrad*proxfact
        circling = away*incircle # [True/False] passed wp,used for flyover as well


        # distance to turn initialisation point [m]
        self.turndist = flyby*np.abs(turnrad *
            np.tan(np.radians(0.5 * np.abs(degto180(qdr%360. - next_qdr%360.)))))


        # Check whether shift based dist is required, set closer than WP turn distance
        swreached = np.where(bs.traf.swlnav * ((dist < self.turndist)+circling))[0]

        # Return True/1.0 for a/c where we have reached waypoint
        return swreached
