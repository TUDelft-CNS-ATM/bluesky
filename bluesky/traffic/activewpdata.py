import numpy as np
import bluesky as bs
from bluesky.tools.aero import nm, g0
from bluesky.tools.misc import degto180
from bluesky.core import Entity

class ActiveWaypoint(Entity, replaceable=True):
    def __init__(self):
        super().__init__()
        with self.settrafarrays():
            self.lat        = np.array([])    # [deg] Active WP latitude
            self.lon        = np.array([])    # [deg] Active WP longitude
            self.nextaltco  = np.array([])    # [m] Altitude to arrive at after distance xtoalt
            self.xtoalt     = np.array([])    # [m] Distance to next altitude constraint
            self.nextspd    = np.array([])    # [CAS[m/s]/Mach] save speed from next wp for next leg
            self.spd        = np.array([])    # [CAS[m/s]/Mach]Active WP speed (constraint or calculated)
            self.spdcon     = np.array([])    # [CAS[m/s]/Mach]Active WP speed constraint
            self.vs         = np.array([])    # [m/s] Active vertical speed to use
            self.turndist   = np.array([])    # [m] Distance when to turn to next waypoint
            self.flyby      = np.array([])    # Flyby switch, when False, flyover (turndist=0.0)
            self.flyturn    = np.array([])    # Flyturn switch, when False, when Fkase, use flyby/flyover
            self.turnrad    = np.array([])    # Flyturn turn radius (<0 => not specified)
            self.turnspd    = np.array([])    # [m/s, CAS] Flyturn turn speed for next turn (<=0 => not specified)
            self.oldturnspd = np.array([])    # [TAS, m/s] Flyturn turn speed for previous turn (<=0 => not specified)
            self.turnfromlastwp = np.array([]) # Currently in flyturn-mode from last waypoint (old turn, beginning of leg)
            self.turntonextwp =  np.array([])  # Currently in flyturn-mode to next waypoint (new flyturn mode, end of leg)
            self.torta      = np.array([])    # [s] NExt req Time of Arrival (RTA) (-999. = None)
            self.xtorta     = np.array([])    # [m] distance ot next RTA
            self.next_qdr   = np.array([])    # [deg] track angle of next leg

    def create(self, n=1):
        super().create(n)
        # LNAV route navigation
        self.lat[-n:]        = 89.99    # [deg]Active WP latitude
        self.nextspd[-n:]    = -999.    # [CAS[m/s]/Mach]Next leg speed from current WP
        self.spd[-n:]        = -999.    # [CAS[m/s]/Mach]Active WP speed
        self.spdcon[-n:]     = -999.    # [CAS[m/s]/Mach]Active WP speed constraint
        self.turndist[-n:]   = 1.0      # [m] Distance to active waypoint where to turn
        self.flyby[-n:]      = 1.0      # Flyby/fly-over switch
        self.flyturn[-n:]    = False    # Flyturn switch, when False, when False, use flyby/flyover
        self.turnrad[-n:]    = -999.    # Flyturn turn radius (<0 => not specified)
        self.turnspd[-n:]    = -999.    # Flyturn turn speed (<0 => not specified)
        self.oldturnspd[-n:] = -999.    # [TAS, m/s] Flyturn turn speed for previous turn (<=0 => not specified)
        self.turnfromlastwp[-n:] = False # Currently in flyturn-mode from last waypoint (old turn, beginning of leg)
        self.turntonextwp[-n:] = False  # Currently in flyturn-mode to next waypoint (new flyturn mode, end of leg)
        self.torta[-n:]      = -999.0   # [s] Req Time of Arrival (RTA) for next wp (-999. = None)
        self.xtorta[-n:]     = 0.0      # Distance to next RTA
        self.next_qdr[-n:]   = -999.0   # [deg] bearing next leg

    def Reached(self, qdr, dist, flyby, flyturn, turnradnm):
        # Calculate distance before waypoint where to start the turn
        # Note: this is a vectorized function, called with numpy arrays
        # It returns the indices where the Reached criterion is True
        #
        # Turn radius:      R = V2 tan phi / g
        # Distance to turn: wpturn = R * tan (1/2 delhdg) but max 4 times radius
        # using default bank angle per flight phase

        # First calculate turn distance
        next_qdr = np.where(self.next_qdr < -900., qdr, self.next_qdr)
        flybyturndist,turnrad = self.calcturn(bs.traf.tas,bs.traf.bank,qdr,next_qdr,turnradnm)

        self.turndist = np.logical_or(flyby,flyturn)*flybyturndist

        # Avoid circling by checking for flying away on almost straight legs with small turndist
        # difference between direction to and track larger than 90
        # but avoid switching wayspoint when trk undefined due to standing still (groundspeed (<1 m/s)
        away  = (np.abs(bs.traf.gs)>1)*(np.abs(degto180(bs.traf.trk%360. - qdr%360.)) > 90.) # difference large than 90


        # Ratio between distance close enough to switch to next wp when flying away
        # When within pro1 nm and flying away: switch also
        proxfact = 1.02 # Turnradius scales this contant , factor => [turnrad]
        incircle = dist<turnrad*proxfact
        circling = away*incircle # [True/False] passed wp,used for flyover as well

        # Check whether shift based dist is required, set closer than WP turn distance
        # Detect indices
        swreached = np.where(bs.traf.swlnav * np.logical_or(away,np.logical_or(dist < self.turndist,circling)))[0]

        # Return indices for which condition is True/1.0 for a/c where we have reached waypoint
        return swreached

    # Calculate turn distance for array or scalar
    def calcturn(self,tas,bank,wpqdr,next_wpqdr,turnradnm=-999.):
        """Calculate distance to wp where to start turn and turn radius in meters"""

        # Calculate turn radius using current speed or use specified turnradius
        turnrad = np.where(turnradnm+0.*tas<0.,
                           tas * tas / (np.maximum(0.01, np.tan(bank)) * g0),
                           turnradnm*nm)

        # turndist is in meters
        turndist = np.abs(turnrad *
           np.tan(np.radians(0.5 * np.abs(degto180(wpqdr%360. - next_wpqdr%360.)))))

        return turndist,turnrad
