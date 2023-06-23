import numpy as np
import bluesky as bs
from bluesky.tools.aero import nm, g0
from bluesky.tools.misc import degto180
from bluesky.core import Entity


class ActiveWaypoint(Entity, replaceable=True):
    def __init__(self):
        super().__init__()
        with self.settrafarrays():
            self.lat         = np.array([])    # [deg] Active WP latitude
            self.lon         = np.array([])    # [deg] Active WP longitude
            self.nextturnlat = np.array([])    # [deg] Next turn WP latitude
            self.nextturnlon = np.array([])    # [deg] Next turn WP longitude
            self.nextturnspd = np.array([])    # [m/s] Next turn WP speed
            self.nextturnbank= np.array([])    # [deg] Next turn WP bank angle
            self.nextturnrad = np.array([])    # [m]   Next turn WP turn radius
            self.nextturnhdgr= np.array([])  # [deg/s] Next turn WP heading rate (<0 => not specified)
            self.nextturnidx = np.array([])    # [-]   Next turn WP index
            self.nextaltco   = np.array([])    # [m] Altitude to arrive at after distance xtoalt
            self.xtoalt      = np.array([])    # [m] Distance to next altitude constraint
            self.nextspd     = np.array([])    # [CAS[m/s]/Mach] save speed from next wp for next leg
            self.spd         = np.array([])    # [CAS[m/s]/Mach]Active WP speed (constraint or calculated)
            self.spdcon      = np.array([])    # [CAS[m/s]/Mach]Active WP speed constraint
            self.vs          = np.array([])    # [m/s] Active vertical speed to use
            self.turndist    = np.array([])    # [m] Distance when to turn to next waypoint
            self.flyby       = np.array([])    # Flyby switch, when False, flyover (turndist=0.0)
            self.flyturn     = np.array([])    # Flyturn switch, customised turn parameters; when False, use flyby/flyover
            self.turnbank    = np.array([])    # Flyturn turn bank angle [deg]
            self.turnrad     = np.array([])    # Flyturn turn radius (<0 => not specified)
            self.turnspd     = np.array([])    # [m/s, CAS] Flyturn turn speed for next turn (<=0 => not specified)
            self.turnhdgr    = np.array([])    # [deg/s]Flyturn turn heading rate (<0 => not specified)
            self.oldturnspd  = np.array([])    # [TAS, m/s] Flyturn turn speed for previous turn (<=0 => not specified)
            self.turnfromlastwp = np.array([]) # Currently in flyturn-mode from last waypoint (old turn, beginning of leg)
            self.turntonextwp =  np.array([])  # Currently in flyturn-mode to next waypoint (new flyturn mode, end of leg)
            self.torta       = np.array([])    # [s] Next req Time of Arrival (RTA) (-999. = None)
            self.xtorta      = np.array([])    # [m] distance to next RTA
            self.next_qdr    = np.array([])    # [deg] track angle of next leg
            self.swlastwp    = np.array([],dtype=bool) # switch indicating this is the last waypoint
            self.curlegdir   = np.array([])    # [deg] direction to active waypoint upon activation
            self.curleglen   = np.array([])    # [deg] direction to active waypoint upon activation

    def create(self, n=1):
        super().create(n)
        # LNAV route navigation
        self.lat[-n:]        = 0.       # [deg]Active WP latitude
        self.lon[-n:]        = 0.       # [deg]Active WP longitude
        self.nextturnlat[-n:]= 0        # [deg] Next turn WP latitude
        self.nextturnlon[-n:]= 0        # [deg] Next turn WP longitude
        self.nextturnspd[-n:]= -999.    # [m/s] Next turn WP speed
        self.nextturnbank[-n:] = -999.  # [deg] Next turb WP bank angle
        self.nextturnrad[-n:]= -999.    # [m]   Next turn WP radius
        self.nextturnhdgr[-n:]= -999.   # [deg/s] Next turn WP heading rate (<0 => not specified)
        self.nextturnidx[-n:]= -999.    # [-] Next turn WP index
        self.nextaltco[-n]   = -999.    # [m] Altitude to arrive at after distance xtoalt
        self.xtoalt[-n:]     = 0.0      # [m] Distance to next altitude constraint
        self.nextspd[-n:]    = -999.    # [CAS[m/s]/Mach]Next leg speed from current WP
        self.spd[-n:]        = -999.    # [CAS[m/s]/Mach]Active WP speed
        self.spdcon[-n:]     = -999.    # [CAS[m/s]/Mach]Active WP speed constraint
        self.turndist[-n:]   = 1.0      # [m] Distance to active waypoint where to turn
        self.flyby[-n:]      = 1.0      # Flyby/fly-over switch
        self.flyturn[-n:]    = False    # Flyturn switch, when False, when False, use flyby/flyover
        self.turnbank[-n:]   = -999.    # Flyturn turn bank angle [deg]
        self.turnrad[-n:]    = -999.    # [m] Flyturn turn radius (<0 => not specified)
        self.turnspd[-n:]    = -999.    # [m/s]Flyturn turn speed (<0 => not specified)
        self.turnhdgr[-n:]   = -999.    # [deg/s]Flyturn turn heading rate (<0 => not specified)
        self.oldturnspd[-n:] = -999.    # [TAS, m/s] Flyturn turn speed for previous turn (<=0 => not specified)
        self.turnfromlastwp[-n:] = False # Currently in flyturn-mode from last waypoint (old turn, beginning of leg)
        self.turntonextwp[-n:] = False  # Currently in flyturn-mode to next waypoint (new flyturn mode, end of leg)
        self.torta[-n:]      = -999.0   # [s] Req Time of Arrival (RTA) for next wp (-999. = None)
        self.xtorta[-n:]     = 0.0      # Distance to next RTA
        self.next_qdr[-n:]   = -999.0   # [deg] bearing next leg
        self.swlastwp[-n:]   = False    # Switch indicating active waypoint is last waypoint
        self.curlegdir[-n:]  = -999.0   # [deg] direction to active waypoint upon activation
        self.curleglen[-n:]  = -999.0   # [nm] distance to active waypoint upon activation
  
    def reached(self, qdr, dist):
        # Calculate distance before waypoint where to start the turn
        # Note: this is a vectorized function, called with numpy traffic arrays
        # It returns the indices where the Reached criterion is True
        #
        # Turn radius:      R = V2 tan phi / g
        # Distance to turn: wpturn = R * tan (1/2 delhdg) but max 4 times radius
        # using default bank angle per flight phase
        # Gather required data
        flyby = self.flyby
        flyturn = self.flyturn

        # Turn dist is zero for flyover, and it is previously calculated in autopilot for others
        self.turndist = np.logical_or(flyby,flyturn)*self.turndist

        # Avoid circling by checking too close to waypoint based on ground speed, assumption using vicinity criterion:
        # flying away and within 4 sec distance based on ground speed (4 sec = sensitivity tuning parameter)

        close2wp = dist/(np.maximum(0.0001,np.abs(bs.traf.gs)))<4.0 # Waypoint is within 4 seconds flight time
        tooclose2turn = close2wp*(np.abs(degto180(bs.traf.trk % 360. - qdr % 360.)) > 90.)

        # When too close to waypoint or we have passed the active waypoint, based on leg direction,switch active waypoint
        # was:  away  = np.logical_or(close2wp,swlastwp)*(np.abs(degto180(bs.traf.trk%360. - qdr%360.)) > 90.) # difference large than 90
        awayorpassed =  np.logical_or(tooclose2turn,np.abs(degto180(qdr-bs.traf.actwp.curlegdir))>90.)

        # Should no longer be needed with leg direction
        # Ratio between distance close enough to switch to next wp when flying away
        # When within pro1 nm and flying away: switch also
        #proxfact = 1.02 # Turnradius scales this contant , factor => [turnrad]
        #incircle = dist<turnrad*proxfact
        #circling = away*incircle # [True/False] passed wp,used for flyover as well

        # Check whether shift based dist is required, set closer than WP turn distance
        # Detect indices
        #swreached = np.where(bs.traf.swlnav * np.logical_or(awayorpassed,np.logical_or(dist < self.turndist,circling)))[0]
        swreached = np.where(bs.traf.swlnav * np.logical_or(awayorpassed,dist < self.turndist))[0]

        # Return indices for which condition is True/1.0 for a/c where we have reached waypoint
        return swreached

    # Calculate turn distance for scalars
    def calcturn(self, acidx, tas , wpqdr, next_wpqdr, turnbank, turnrad, turnspd, turnhdgr, flyturn, flyby):
        """Calculate the properties of a turn in function of the input. Inputs are SCALARS."""
        # If this is not a flyturn waypoint, just return standard values
        # Calculate the sum of bools to figure out things more easily
        num_defined_values = sum([turnbank>0, turnrad>0, turnspd>0, turnhdgr>0])
        # Case 1: No value is given or not a flyturn, use defaults
        if num_defined_values == 0 or not flyturn:
            turnbank = np.rad2deg(bs.traf.ap.bankdef[acidx])
            turnspd = tas
            # Calculate the radius
            turnrad = turnspd**2/(g0*np.tan(np.deg2rad(turnbank)))
            if not flyby:
                # This is a flyover waypoint, so we have to fly OVER it, so turndist is 0
                return 0, turnrad, turnspd, turnbank, turnhdgr
        # Case 2: only one value is given. We then want to keep the speed as TAS unless the speed is the one that isn't specified
        elif num_defined_values == 1:
            # If the velocity is the given one, take default bank and calculate remaining values
            if turnspd>0:
                turnbank = 25
                turnrad = turnspd**2/(g0*np.tan(np.deg2rad(turnbank)))
            #Otherwise, keep velocity constant and calculate the remaining values
            elif turnrad>0:
                turnspd = tas
                turnbank = np.arctan(turnspd**2/(turnrad * g0))
                
            elif turnbank>0:
                turnspd = tas
                turnrad = turnspd**2 / (g0 * np.tan(np.deg2rad(turnbank)))
                
            elif turnhdgr>0:
                turnspd = tas
                #Calculate the other two
                turnbank = np.arctan(turnhdgr*turnspd/g0)
                turnrad = turnspd / turnhdgr
        # Case 3: We have two defined values and need to calculate the other
        elif num_defined_values == 2:
            # Need to figure out which ones and calculate the remaining
            if turnrad>0 and turnbank>0:
                turnspd = np.sqrt(g0*turnrad*np.tan(np.deg2rad(turnbank)))
            elif turnrad>0 and turnspd>0:
                turnbank = np.arctan(turnspd**2/(turnrad*g0))
            elif turnspd>0 and turnbank>0:
                turnrad = turnspd**2/(g0*np.tan(np.deg2rad(turnbank)))
            # Now the cases where one of the known vars is heading rate
            elif turnhdgr>0 and turnspd>0:
                turnbank = np.arctan(turnhdgr*turnspd/g0)
                turnrad = turnspd / turnhdgr
            elif turnhdgr>0 and turnbank>0:
                turnspd = g0*np.tan(np.deg2rad(turnbank))/turnhdgr
                turnrad = turnspd / turnhdgr
            elif turnhdgr>0 and turnrad>0:
                turnspd = turnhdgr * turnrad
                turnbank = np.arctan(turnhdgr*turnspd/g0)
        
        # turndist is in meters
        turndist = np.abs(turnrad * np.tan(np.radians(0.5 * np.abs(degto180(wpqdr%360. - next_wpqdr%360.)))))
        return turndist, turnrad, turnspd, turnbank, turnhdgr
