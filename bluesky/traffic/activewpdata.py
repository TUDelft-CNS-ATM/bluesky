import numpy as np
import bluesky as bs
from bluesky.tools.trafficarrays import TrafficArrays, RegisterElementParameters
from bluesky.tools.aero import nm, g0
from bluesky.tools.misc import degto180


class ActiveWaypoint(TrafficArrays):
    def __init__(self):
        super(ActiveWaypoint, self).__init__()
        with RegisterElementParameters(self):
            self.lat       = np.array([])  # [deg] Active WP latitude
            self.lon       = np.array([])  # [deg] Active WP longitude
            self.nextaltco = np.array([])  # [m] Altitude to arrive at after distance xtoalt
            self.xtoalt    = np.array([])  # [m] Distance to next altitude constraint
            self.spd       = np.array([])  # [CAS[m/s]/Mach]Active WP speed
            self.vs        = np.array([])  # [m/s] Active vertical speed to use
            self.turndist  = np.array([])  # [m] Distance when to turn to next waypoint
            self.flyby     = np.array([])  # Flyby switch, when False, flyover (turndist=0.0)
            self.next_qdr  = np.array([])  # [deg] track angle of next leg

    def create(self, n=1):
        super(ActiveWaypoint, self).create(n)
        # LNAV route navigation
        self.lat[-n:]       = 89.99    # [deg]Active WP latitude
        self.spd[-n:]       = -999.    # [CAS[m/s]/Mach]Active WP speed
        self.turndist[-n:]  = 1.0      # [m] Distance to active waypoint where to turn
        self.flyby[-n:]     = 1.0      # Flyby/fly-over switch
        self.next_qdr[-n:]  = -999.0   # [deg] bearing next leg

    def Reached(self, qdr, dist, flyby):
        # Calculate distance before waypoint where to start the turn
        # Turn radius:      R = V2 tan phi / g
        # Distance to turn: wpturn = R * tan (1/2 delhdg) but max 4 times radius
        # using default bank angle per flight phase

        # First calculate turn diatance
        next_qdr = np.where(self.next_qdr < -900., qdr, self.next_qdr)
        flybyturndist,turnrad = self.calcturn(bs.traf.tas,bs.traf.bank,qdr,next_qdr)
        self.turndist = flyby*flybyturndist

        # Avoid circling by checking for flying away
        away     = np.abs(degto180(bs.traf.trk%360. - qdr%360.)) > 90. # difference large than 90

        # Ratio between distance close enough to switch to next wp when flying away
        # When within pro1 nm and flying away: switch also
        proxfact = 1.01 # Turnradius scales this contant , factor => [turnrad]
        incircle = dist<turnrad*proxfact
        circling = away*incircle # [True/False] passed wp,used for flyover as well

        # Check whether shift based dist is required, set closer than WP turn distance
        swreached = np.where(bs.traf.swlnav * ((dist < self.turndist)+circling))[0]

        # Return True/1.0 for a/c where we have reached waypoint
        return swreached

    # Calculate turn distance for array or scalar
    def calcturn(self,tas,bank,wpqdr,next_wpqdr):
        # Turn radius in meters!
        turnrad = tas * tas / (np.maximum(0.01, np.tan(bank)) * g0)

        # turndist in meters
        turndist = np.abs(turnrad *
           np.tan(np.radians(0.5 * np.abs(degto180(wpqdr%360. - next_wpqdr%360.)))))

        return turndist,turnrad
