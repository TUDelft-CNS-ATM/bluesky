""" Autopilot Implementation."""
from math import sin, cos, radians, sqrt, atan
import numpy as np
try:
    from collections.abc import Collection
except ImportError:
    # In python <3.3 collections.abc doesn't exist
    from collections import Collection
import bluesky as bs
from bluesky import stack
from bluesky.tools import geo
from bluesky.tools.misc import degto180
from bluesky.tools.position import txt2pos
from bluesky.tools.aero import ft, nm, fpm, vcasormach2tas, vcas2tas, tas2cas, cas2tas, g0
from bluesky.core import Entity, timed_function
from .route import Route

#debug
from inspect import stack as callstack
from bluesky.tools.datalog import crelog

bs.settings.set_variable_defaults(fms_dt=10.5)


class Autopilot(Entity, replaceable=True):
    ''' BlueSky Autopilot implementation. '''
    def __init__(self):
        super().__init__()

        # Standard self.steepness for descent
        self.steepness = 3000. * ft / (10. * nm)

        # From here, define object arrays
        with self.settrafarrays():

            # FMS directions
            self.trk = np.array([])
            self.spd = np.array([])
            self.tas = np.array([])
            self.alt = np.array([])
            self.vs  = np.array([])

            # VNAV variables
            self.swtoc    = np.array([])  # ToC switch to switch on VNAV Top of Climb logic (default value True)
            self.swtod    = np.array([])  # ToD switch to switch on VNAV Top of Descent logic (default value True)

            self.dist2vs  = np.array([])  # distance from coming waypoint to TOD
            self.dist2accel = np.array([]) # Distance to go to acceleration(decelaration) for turn next waypoint [nm]

            self.swvnavvs = np.array([])  # whether to use given VS or not
            self.vnavvs   = np.array([])  # vertical speed in VNAV


            # LNAV variables
            self.qdr2wp      = np.array([]) # Direction to waypoint from the last time passing was checked
                                            # to avoid 180 turns due to updated qdr shortly before passing wp
            self.dist2wp     = np.array([]) # [m] Distance to active waypoint
            self.qdrturn     = np.array([]) # qdr to next turn]
            self.dist2turn   = np.array([]) # Distance to next turn [m]

            self.inturn = np.array([]) # If we're in a turn maneuver or not

            # Traffic navigation information
            self.orig = []  # Four letter code of origin airport
            self.dest = []  # Four letter code of destination airport

            # Default values
            self.bankdef = np.array([])  # nominal bank angle, [radians]
            self.vsdef = np.array([]) # [m/s]default vertical speed of autopilot
            
            # Currently used roll/bank angle [rad]
            self.turnphi = np.array([])  # [rad] bank angle setting of autopilot

            # Route objects
            self.route = []


        self.idxreached = []    # List indices of aircraft who have reached their active waypoint

    def create(self, n=1):
        super().create(n)

        # FMS directions
        self.trk[-n:] = bs.traf.trk[-n:]
        self.tas[-n:] = bs.traf.tas[-n:]
        self.alt[-n:] = bs.traf.alt[-n:]
        self.vs[-n:]  = -999

        # Default ToC/ToD logic on
        self.swtoc[-n:] = True
        self.swtod[-n:] = True

        # VNAV Variables
        self.dist2vs[-n:] = -999.
        self.dist2accel[-n:] = -999.  # Distance to go to acceleration(decelaration) for turn next waypoint [nm]


        # LNAV variables
        self.qdr2wp[-n:] = -999.   # Direction to waypoint from the last time passing was checked
        self.dist2wp[-n:]  = -999. # Distance to go to next waypoint [nm]

        # Traffic performance data
        #(temporarily default values)
        self.vsdef[-n:] = 1500. * fpm   # default vertical speed of autopilot
        self.bankdef[-n:] = np.radians(25.)

        # Route objects
        for ridx, acid in enumerate(bs.traf.id[-n:]):
            self.route[ridx - n] = Route(acid)


    #no longer timed @timed_function(name='fms', dt=bs.settings.fms_dt, manual=True)
    def wppassingcheck(self, qdr, dist): # qdr [deg], dist [m[
        """
        The actwp is the interface between the list of waypoint data in the route object and the autopilot guidance
        when LNAV is on (heading) and optionally VNAV is on (spd & altitude)

        actwp data contains traffic arrays, to allow vectorizing the guidance logic.

        Waypoint switching (just like the adding, deletion in route) are event driven commands and
        therefore not vectorized as they occur rarely compared to the guidance.

        wppassingcheck contains the waypoint switching function:
        - Check which aircraft i have reached their active waypoint
        - Reached function return list of indices where reached logic is True
        - Get the waypoint data to the actwp (active waypoint data)
        - Shift waypoint (last,next etc.) data for aircraft i where necessary
        - Shift and maintain data (see last- and next- prefix in varubale name) e.g. to continue a special turn
        - Prepare some VNAV triggers along the new leg for the VNAV profile (where to start descent/climb)
        """

        # Get list of indices of aircraft which have reached their active waypoint
        # This vectorized function checks the passing of the waypoint using a.o. the current turn radius
        self.idxreached = bs.traf.actwp.Reached(qdr, dist, bs.traf.actwp.flyby,
                                       bs.traf.actwp.flyturn,bs.traf.actwp.turnrad,bs.traf.actwp.swlastwp)

        # For the one who have reached their active waypoint, update vectorized leg data for guidance
        for i in self.idxreached:

            #debug commands to check VNAV state while passing waypoint
            #print("Passing waypoint",bs.traf.ap.route[i].wpname[bs.traf.ap.route[i].iactwp])
            #print("dist2wp,dist2vs",self.dist2wp[i]/nm,self.dist2vs[i]/nm) # distance to wp & distance to ToD/ToC

            # Save current wp speed for use on next leg when we pass this waypoint
            # VNAV speeds are always FROM-speeds, so we accelerate/decellerate at the waypoint
            # where this speed is specified, so we need to save it for use now
            # before getting the new data for the next waypoint

            # Get speed for next leg from the waypoint we pass now and set as active spd
            bs.traf.actwp.spd[i]    = bs.traf.actwp.nextspd[i]
            bs.traf.actwp.spdcon[i] = bs.traf.actwp.nextspd[i]

            # Execute stack commands for the still active waypoint, which we pass
            self.route[i].runactwpstack()

            # Special turns: specified by turn radius or bank angle
            # If specified, use the given turn radius of passing wp for bank angle
            if bs.traf.actwp.flyturn[i]:
                if bs.traf.actwp.turnspd[i]>=0.:
                    turnspd = bs.traf.actwp.turnspd[i]
                else:
                    turnspd = bs.traf.tas[i]

                if bs.traf.actwp.turnrad[i] > 0.:
                    self.turnphi[i] = atan(turnspd*turnspd/(bs.traf.actwp.turnrad[i]*nm*g0)) # [rad]
                else:
                    self.turnphi[i] = 0.0  # [rad] or leave untouched???

            else:
                self.turnphi[i] = 0.0  #[rad] or leave untouched???

            # Get next wp, if there still is one
            if not bs.traf.actwp.swlastwp[i]:
                lat, lon, alt, bs.traf.actwp.nextspd[i], \
                bs.traf.actwp.xtoalt[i], toalt, \
                    bs.traf.actwp.xtorta[i], bs.traf.actwp.torta[i], \
                    lnavon, flyby, flyturn, turnrad, turnspd,\
                    bs.traf.actwp.next_qdr[i], bs.traf.actwp.swlastwp[i] =      \
                    self.route[i].getnextwp()  # [m] note: xtoalt,nextaltco are in meters


                bs.traf.actwp.nextturnlat[i], bs.traf.actwp.nextturnlon[i], \
                bs.traf.actwp.nextturnspd[i], bs.traf.actwp.nextturnrad[i], \
                bs.traf.actwp.nextturnidx[i] = self.route[i].getnextturnwp()

            else:
                # Prevent trying to activate the next waypoint when it was already the last waypoint
                # In case of end of route/no more waypoints: switch off LNAV using the lnavon
                bs.traf.swlnav[i] = False
                bs.traf.swvnav[i] = False
                bs.traf.swvnavspd[i] = False
                continue # Go to next a/c which reached its active waypoint

            # Check LNAV switch returned by getnextwp
            # Switch off LNAV if it failed to get next wpdata
            if not lnavon and bs.traf.swlnav[i]:
                bs.traf.swlnav[i] = False
                # Last wp: copy last wp values for alt and speed in autopilot
                if bs.traf.swvnavspd[i] and bs.traf.actwp.nextspd[i]>= 0.0:
                    bs.traf.selspd[i] = bs.traf.actwp.nextspd[i]

            # In case of no LNAV, do not allow VNAV mode to be active
            bs.traf.swvnav[i] = bs.traf.swvnav[i] and bs.traf.swlnav[i]

            bs.traf.actwp.lat[i] = lat  # [deg]
            bs.traf.actwp.lon[i] = lon  # [deg]
            # 1.0 in case of fly by, else fly over
            bs.traf.actwp.flyby[i] = int(flyby)

            # Update qdr and turndist for this new waypoint for ComputeVNAV
            qdr[i], distnmi = geo.qdrdist(bs.traf.lat[i], bs.traf.lon[i],
                                          bs.traf.actwp.lat[i], bs.traf.actwp.lon[i])

            #dist[i] = distnmi * nm
            self.dist2wp[i] = distnmi*nm

            bs.traf.actwp.curlegdir[i] = qdr[i]
            bs.traf.actwp.curleglen[i] = self.dist2wp[i]

            # User has entered an altitude for the new waypoint
            if alt >= -0.01: # positive alt on this waypoint means altitude constraint
                bs.traf.actwp.nextaltco[i] = alt  # [m]
                bs.traf.actwp.xtoalt[i] = 0.0
            else:
                bs.traf.actwp.nextaltco[i] = toalt  # [m]

            #if not bs.traf.swlnav[i]:
            #    bs.traf.actwp.spd[i] = -997.

            # VNAV spd mode: use speed of this waypoint as commanded speed
            # while passing waypoint and save next speed for passing next wp
            # Speed is now from speed! Next speed is ready in wpdata
            if bs.traf.swvnavspd[i] and bs.traf.actwp.spd[i]>= 0.0:
                    bs.traf.selspd[i] = bs.traf.actwp.spd[i]

            # Update turndist so ComputeVNAV works, is there a next leg direction or not?
            if bs.traf.actwp.next_qdr[i] < -900.:
                local_next_qdr = qdr[i]
            else:
                local_next_qdr = bs.traf.actwp.next_qdr[i]

            # Calculate turn dist (and radius which we do not use) now for scalar variable [i]
            bs.traf.actwp.turndist[i], dummy = \
                bs.traf.actwp.calcturn(bs.traf.tas[i], self.bankdef[i],
                                        qdr[i], local_next_qdr,turnrad)  # update turn distance for VNAV

            # Get flyturn switches and data
            bs.traf.actwp.flyturn[i]     = flyturn
            bs.traf.actwp.turnrad[i]     = turnrad

            # Pass on whether currently flyturn mode:
            # at beginning of leg,c copy tonextwp to lastwp
            # set next turn False
            bs.traf.actwp.turnfromlastwp[i] = bs.traf.actwp.turntonextwp[i]
            bs.traf.actwp.turntonextwp[i]   = False

            # Keep both turning speeds: turn to leg and turn from leg
            bs.traf.actwp.oldturnspd[i]  = bs.traf.actwp.turnspd[i] # old turnspd, turning by this waypoint
            if bs.traf.actwp.flyturn[i]:
                bs.traf.actwp.turnspd[i] = turnspd                  # new turnspd, turning by next waypoint
            else:
                bs.traf.actwp.turnspd[i] = -990.

            # Reduce turn dist for reduced turnspd
            if bs.traf.actwp.flyturn[i] and bs.traf.actwp.turnrad[i]<0.0 and bs.traf.actwp.turnspd[i]>=0.:
                turntas = cas2tas(bs.traf.actwp.turnspd[i], bs.traf.alt[i])
                bs.traf.actwp.turndist[i] = bs.traf.actwp.turndist[i]*turntas*turntas/(bs.traf.tas[i]*bs.traf.tas[i])

            # VNAV = FMS ALT/SPD mode incl. RTA
            self.ComputeVNAV(i, toalt, bs.traf.actwp.xtoalt[i], bs.traf.actwp.torta[i],
                             bs.traf.actwp.xtorta[i])

        

        # End of reached-loop: the per waypoint i switching loop

        # Update qdr2wp with up-to-date qdr, now that we have checked passing wp
        self.qdr2wp = qdr%360.

        # Continuous guidance when speed constraint on active leg is in update-method

        # If still an RTA in the route and currently no speed constraint
        for iac in np.where((bs.traf.actwp.torta > -99.)*(bs.traf.actwp.spdcon<0.0))[0]:
            iwp = bs.traf.ap.route[iac].iactwp
            if bs.traf.ap.route[iac].wprta[iwp]>-99.:

                 # For all a/c flying to an RTA waypoint, recalculate speed more often
                dist2go4rta = geo.kwikdist(bs.traf.lat[iac],bs.traf.lon[iac],
                                           bs.traf.actwp.lat[iac],bs.traf.actwp.lon[iac])*nm \
                               + bs.traf.ap.route[iac].wpxtorta[iwp] # last term zero for active wp rta

                # Set bs.traf.actwp.spd to rta speed, if necessary
                self.setspeedforRTA(iac,bs.traf.actwp.torta[iac],dist2go4rta)

                # If VNAV speed is on (by default coupled to VNAV), use it for speed guidance
                if bs.traf.swvnavspd[iac] and bs.traf.actwp.spd[iac]>=0.0:
                     bs.traf.selspd[iac] = bs.traf.actwp.spd[iac]

    def update(self):
        # FMS LNAV mode:
        # qdr[deg],distinnm[nm]
        qdr, distinnm = geo.qdrdist(bs.traf.lat, bs.traf.lon,
                                    bs.traf.actwp.lat, bs.traf.actwp.lon)  # [deg][nm])

        self.qdr2wp  = qdr
        self.dist2wp = distinnm*nm  # Conversion to meters

        # Check possible waypoint shift. Note: qdr, dist2wp will be updated accordingly in case of wp switch
        self.wppassingcheck(qdr, self.dist2wp) # Updates self.qdr2wp when necessary

        #================= Continuous FMS guidance ========================

        # Note that the code below is vectorized, with traffic arrays, so for all aircraft
        # ComputeVNAV and inside waypoint loop of wppassingcheck, it was scalar (per a/c with index i)

        # VNAV altitude guidance logic (using the variables prepared by ComputeVNAV when activating waypoint)

        # First question is:
        # - Can we please we start to descend or to climb?
        #
        # The variable dist2vs indicates the distance to the active waypoint where we should start our climb/descend
        # Only use this logic if there is a valid next altitude constraint (nextaltco).
        #
        # Well, when Top of Descent (ToD) switch is on, descend as late as possible,
        # But when Top of Climb switch is on or off, climb as soon as possible, only difference is steepness used in ComputeVNAV
        # to calculate bs.traf.actwp.vs

        startdescorclimb = (bs.traf.actwp.nextaltco>=-0.1) * \
                           np.logical_or((bs.traf.alt>bs.traf.actwp.nextaltco) *\
                                         np.logical_or((self.dist2wp < self.dist2vs+bs.traf.actwp.turndist),\
                                                       (np.logical_not(self.swtod))),\
                                         bs.traf.alt<bs.traf.actwp.nextaltco)

        # print("self.dist2vs =",self.dist2vs)

        # If not lnav:Climb/descend if doing so before lnav/vnav was switched off
        #    (because there are no more waypoints). This is needed
        #    to continue descending when you get into a conflict
        #    while descending to the destination (the last waypoint)
        #    Use 0.1 nm (185.2 m) circle in case turndist might be zero
        self.swvnavvs = bs.traf.swvnav * np.where(bs.traf.swlnav, startdescorclimb,\
                                        self.dist2wp <= np.maximum(0.1*nm,bs.traf.actwp.turndist))

        # Recalculate V/S based on current altitude and distance to next alt constraint
        # How much time do we have before we need to descend?
        # Now done in ComputeVNAV
        # See ComputeVNAV for bs.traf.actwp.vs calculation

        self.vnavvs  = np.where(self.swvnavvs, bs.traf.actwp.vs, self.vnavvs)
        #was: self.vnavvs  = np.where(self.swvnavvs, self.steepness * bs.traf.gs, self.vnavvs)

        # self.vs = np.where(self.swvnavvs, self.vnavvs, self.vsdef * bs.traf.limvs_flag)
        # for VNAV use fixed V/S and change start of descent
        selvs = np.where(abs(bs.traf.selvs) > 0.1, bs.traf.selvs, self.vsdef) # m/s
        self.vs  = np.where(self.swvnavvs, self.vnavvs, selvs)
        self.alt = np.where(self.swvnavvs, bs.traf.actwp.nextaltco, bs.traf.selalt)

        # When descending or climbing in VNAV also update altitude command of select/hold mode
        bs.traf.selalt = np.where(self.swvnavvs,bs.traf.actwp.nextaltco,bs.traf.selalt)

        # LNAV commanded track angle
        self.trk = np.where(bs.traf.swlnav, self.qdr2wp, self.trk)

        # FMS speed guidance: anticipate accel/decel distance for next leg or turn

        # Calculate actual distance it takes to decelerate/accelerate based on two cases: turning speed (decel)

        # Normally next leg speed (actwp.spd) but in case we fly turns with a specified turn speed
        # use the turn speed

        # Is turn speed specified and are we not already slow enough? We only decelerate for turns, not accel.
        turntas       = np.where(bs.traf.actwp.nextturnspd>0.0, vcas2tas(bs.traf.actwp.nextturnspd, bs.traf.alt),
                                 -1.0+0.*bs.traf.tas)
        # Switch is now whether the aircraft has any turn waypoints
        swturnspd     = bs.traf.actwp.nextturnidx > 0
        turntasdiff   = np.maximum(0.,(bs.traf.tas - turntas)*(turntas>0.0))

        # t = (v1-v0)/a ; x = v0*t+1/2*a*t*t => dx = (v1*v1-v0*v0)/ (2a)
        dxturnspdchg = distaccel(turntas,bs.traf.tas, bs.traf.perf.axmax)

        # Decelerate or accelerate for next required speed because of speed constraint or RTA speed
        # Note that because nextspd comes from the stack, and can be either a mach number or
        # a calibrated airspeed, it can only be converted from Mach / CAS [kts] to TAS [m/s]
        # once the altitude is known.
        nexttas = vcasormach2tas(bs.traf.actwp.nextspd, bs.traf.alt)
#
        dxspdconchg = distaccel(bs.traf.tas, nexttas, bs.traf.perf.axmax)

        qdrturn, dist2turn = geo.qdrdist(bs.traf.lat, bs.traf.lon,
                                        bs.traf.actwp.nextturnlat, bs.traf.actwp.nextturnlon)

        self.qdrturn = qdrturn
        dist2turn = dist2turn * nm

        # Where we don't have a turn waypoint, as in turn idx is negative, then put distance
        # as Earth circumference.
        self.dist2turn = np.where(bs.traf.actwp.nextturnidx > 0, dist2turn, 40075000)

        # Check also whether VNAVSPD is on, if not, SPD SEL has override for next leg
        # and same for turn logic
        usenextspdcon = (self.dist2wp < dxspdconchg)*(bs.traf.actwp.nextspd>-990.) * \
                            bs.traf.swvnavspd*bs.traf.swvnav*bs.traf.swlnav

        useturnspd = np.logical_or(bs.traf.actwp.turntonextwp,
                                   (self.dist2turn < (dxturnspdchg+bs.traf.actwp.turndist))) * \
                                        swturnspd*bs.traf.swvnavspd*bs.traf.swvnav*bs.traf.swlnav

        # Hold turn mode can only be switched on here, cannot be switched off here (happeps upon passing wp)
        bs.traf.actwp.turntonextwp = bs.traf.swlnav*np.logical_or(bs.traf.actwp.turntonextwp,useturnspd)

        # Which CAS/Mach do we have to keep? VNAV, last turn or next turn?
        oncurrentleg = (abs(degto180(bs.traf.trk - qdr)) < 2.0) # [deg]
        inoldturn    = (bs.traf.actwp.oldturnspd > 0.) * np.logical_not(oncurrentleg)

        # Avoid using old turning speeds when turning of this leg to the next leg
        # by disabling (old) turningspd when on leg
        bs.traf.actwp.oldturnspd = np.where(oncurrentleg*(bs.traf.actwp.oldturnspd>0.), -998.,
                                            bs.traf.actwp.oldturnspd)

        # turnfromlastwp can only be switched off here, not on (latter happens upon passing wp)
        bs.traf.actwp.turnfromlastwp = np.logical_and(bs.traf.actwp.turnfromlastwp,inoldturn)

        # Select speed: turn sped, next speed constraint, or current speed constraint
        bs.traf.selspd = np.where(useturnspd,bs.traf.actwp.nextturnspd,
                                  np.where(usenextspdcon, bs.traf.actwp.nextspd,
                                           np.where((bs.traf.actwp.spdcon>=0)*bs.traf.swvnavspd,bs.traf.actwp.spd,
                                                                            bs.traf.selspd)))

        # Temporary override when still in old turn
        bs.traf.selspd = np.where(inoldturn*(bs.traf.actwp.oldturnspd>0.)*bs.traf.swvnavspd*bs.traf.swvnav*bs.traf.swlnav,
                                  bs.traf.actwp.oldturnspd,bs.traf.selspd)

        self.inturn = np.logical_or(useturnspd,inoldturn)

        # Below crossover altitude: CAS=const, above crossover altitude: Mach = const
        self.tas = vcasormach2tas(bs.traf.selspd, bs.traf.alt)

    def ComputeVNAV(self, idx, toalt, xtoalt, torta, xtorta):
        """
        This function to do VNAV (and RTA) calculations is only called only once per leg for one aircraft idx.
        If:
         - switching to next waypoint
         - when VNAV is activated
         - when a DIRECT is given

        It prepares the profile of this leg using the the current altitude and the next altitude constraint (nextaltco).
        The distance to the next altitude constraint is given by xtoalt [m] after active waypoint.

        Options are (classic VNAV logic, swtoc and swtod True):
        - no altitude constraint in the future, do nothing
        - Top of CLimb logic (swtoc=True): if next altitude constrain is baove us, climb as soon as possible with default steepness
        - Top of Descent Logic (swtod =True) Use ToD logic: descend as late aspossible, based on
          steepness. Prepare a ToD somewhere on the leg if necessary based on distance to next altitude constraint.
          This is done by calculating distance to next waypoint where descent should start

        Alternative logic (e.g. for UAVs or GA):
        - swtoc=False and next alt co is above us, climb with the angle/steepness needed to arrive at the altitude at
        the waypoint with the altitude constraint (xtoalt m after active waypoint)
        - swtod=False and next altco is below us, descend with the angle/steepness needed to arrive at at the altitude at
        the waypoint with the altitude constraint (xtoalt m after active waypoint)

        Output if this function:
        self.dist2vs = distance 2 next waypoint where climb/descent needs to activated
        bs.traf.actwp.vs =  V/S to be used during climb/descent part, so when dist2wp<dist2vs [m] (to next waypoint)
        """

        #print ("ComputeVNAV for",bs.traf.id[idx],":",toalt/ft,"ft  ",xtoalt/nm,"nm")
        #print("Called by",callstack()[1].function)

        # Check  whether active waypoint speed needs to be adjusted for RTA
        # sets bs.traf.actwp.spd, if necessary
        # debug print("xtorta+legdist =",(xtorta+legdist)/nm)
        self.setspeedforRTA(idx, torta, xtorta + self.dist2wp[idx])  # all scalar

        # Check if there is a target altitude and VNAV is on, else return doing nothing
        if toalt < 0 or not bs.traf.swvnav[idx]:
            self.dist2vs[idx] = -999999. #dist to next wp will never be less than this, so VNAV will do nothing
            return

        # So: somewhere there is an altitude constraint ahead
        # Compute proper values for bs.traf.actwp.nextaltco, self.dist2vs, self.alt, bs.traf.actwp.vs
        # Descent VNAV mode (T/D logic)
        #
        # xtoalt  =  distance to go to next altitude constraint at a waypoint in the route
        #            (could be beyond next waypoint) [m]
        #
        # toalt   = altitude at next waypoint with an altitude constraint
        #
        # dist2vs = autopilot starts climb or descent when the remaining distance to next waypoint
        #           is this distance
        #
        #
        # VNAV Guidance principle:
        #
        #
        #                          T/C------X---T/D
        #                           /    .        \
        #                          /     .         \
        #       T/C----X----.-----X      .         .\
        #       /           .            .         . \
        #      /            .            .         .  X---T/D
        #     /.            .            .         .        \
        #    / .            .            .         .         \
        #   /  .            .            .         .         .\
        # pos  x            x            x         x         x X
        #
        #
        #  X = waypoint with alt constraint  x = Wp without prescribed altitude
        #
        # - Ignore and look beyond waypoints without an altitude constraint
        # - Climb as soon as possible after previous altitude constraint
        #   and climb as fast as possible, so arriving at alt earlier is ok
        # - Descend at the latest when necessary for next altitude constraint
        #   which can be many waypoints beyond current actual waypoint
        epsalt = 2.*ft # deadzone
        #
        if bs.traf.alt[idx] > toalt + epsalt:
            # Stop potential current climb (e.g. due to not making it to previous altco)
            # then stop immediately, as in: do not make it worse.
            if bs.traf.vs[idx]>0.0001:
                self.vnavvs[idx] = 0.0
                self.alt[idx] = bs.traf.alt[idx]
                if bs.traf.swvnav[idx]:
                    bs.traf.selalt[idx] = bs.traf.alt[idx]

            # Descent modes: VNAV (= swtod/Top of Descent logic) or aiming at next alt constraint

            # Calculate max allowed altitude at next wp (above toalt)
            bs.traf.actwp.nextaltco[idx] = toalt  # [m] next alt constraint
            bs.traf.actwp.xtoalt[idx]    = xtoalt # [m] distance to next alt constraint measured from next waypoint


            # VNAV ToD logic
            if self.swtod[idx]:
                # Get distance to waypoint
                self.dist2wp[idx] = nm*geo.kwikdist(bs.traf.lat[idx], bs.traf.lon[idx],
                                                 bs.traf.actwp.lat[idx],
                                                 bs.traf.actwp.lon[idx])  # was not always up to date, so update first

                # Distance to next waypoint where we need to start descent (top of descent) [m]
                descdist = abs(bs.traf.alt[idx] - toalt) / self.steepness  # [m] required length for descent, uses default steepness!
                self.dist2vs[idx] = descdist - xtoalt   # [m] part of that length on this leg

                #print(bs.traf.id[idx],"traf.alt =",bs.traf.alt[idx]/ft,"ft toalt = ",toalt/ft,"ft descdist =",descdist/nm,"nm")
                #print ("d2wp = ",self.dist2wp[idx]/nm,"nm d2vs = ",self.dist2vs[idx]/nm,"nm")
                #print("xtoalt =",xtoalt/nm,"nm descdist =",descdist/nm,"nm")

                # Exceptions: Descend now?
                #print("Active WP:",bs.traf.ap.route[idx].wpname[bs.traf.ap.route[idx].iactwp])
                #print("dist2wp,turndist, dist2vs= ",self.dist2wp[idx],bs.traf.actwp.turndist[idx],self.dist2vs[idx])
                if self.dist2wp[idx] - 1.02*bs.traf.actwp.turndist[idx] < self.dist2vs[idx]:  # Urgent descent, we're late![m]
                    # Descend now using whole remaining distance on leg to reach altitude
                    self.alt[idx] = bs.traf.actwp.nextaltco[idx]  # dial in altitude of next waypoint as calculated
                    t2go = self.dist2wp[idx]/max(0.01,bs.traf.gs[idx])
                    bs.traf.actwp.vs[idx] = (bs.traf.alt[idx]-toalt)/max(0.01,t2go)

                elif xtoalt<descdist: # Not on this leg, no descending is needed at next waypoint
                    # Top of decent needs to be on this leg, as next wp is in descent
                    bs.traf.actwp.vs[idx] = -abs(self.steepness) * (bs.traf.gs[idx] +
                                                                    (bs.traf.gs[idx] < 0.2 * bs.traf.tas[idx]) *
                                                                    bs.traf.tas[idx])

                else:
                    # else still level
                    bs.traf.actwp.vs[idx] = 0.0

            else:

                # We are higher but swtod = False, so there is no ToD descent logic, simply aim at next altco
                steepness_ = (bs.traf.alt[idx]-bs.traf.actwp.nextaltco[idx])/(max(0.01,self.dist2wp[idx]+xtoalt))
                bs.traf.actwp.vs[idx] = -abs(steepness_) * (bs.traf.gs[idx] +
                                                           (bs.traf.gs[idx] < 0.2 * bs.traf.tas[idx]) * bs.traf.tas[
                                                               idx])
                self.dist2vs[idx]      = 99999. #[m] Forces immediate descent as current distance to next wp will be less

                # print("in else swtod for ", bs.traf.id[idx])

        # VNAV climb mode: climb as soon as possible (T/C logic)
        elif bs.traf.alt[idx] < toalt - 9.9 * ft:
            # Stop potential current descent (e.g. due to not making it to previous altco)
            # then stop immediately, as in: do not make it worse.
            if bs.traf.vs[idx] < -0.0001:
                self.vnavvs[idx] = 0.0
                self.alt[idx] = bs.traf.alt[idx]
                if bs.traf.swvnav[idx]:
                    bs.traf.selalt[idx] = bs.traf.alt[idx]

            # Altitude we want to climb to: next alt constraint in our route (could be further down the route)
            bs.traf.actwp.nextaltco[idx] = toalt   # [m]
            bs.traf.actwp.xtoalt[idx]    = xtoalt  # [m] distance to next alt constraint measured from next waypoint
            self.alt[idx]          = bs.traf.actwp.nextaltco[idx]  # dial in altitude of next waypoint as calculated
            self.dist2vs[idx]      = 99999. #[m] Forces immediate climb as current distance to next wp will be less

            t2go = max(0.1, self.dist2wp[idx]+xtoalt) / max(0.01, bs.traf.gs[idx])
            if self.swtoc[idx]:
                steepness_ = self.steepness # default steepness
            else:
                steepness_ = (bs.traf.alt[idx] - bs.traf.actwp.nextaltco[idx]) / (max(0.01, self.dist2wp[idx] + xtoalt))

            bs.traf.actwp.vs[idx]  = np.maximum(steepness_*bs.traf.gs[idx],
                                       (bs.traf.actwp.nextaltco[idx] - bs.traf.alt[idx]) / t2go) # [m/s]
        # Level leg: never start V/S
        else:
            self.dist2vs[idx] = -999.  # [m]

        return

    def setspeedforRTA(self, idx, torta, xtorta):
        #debug print("setspeedforRTA called, torta,xtorta =",torta,xtorta/nm)

        # Calculate required CAS to meet RTA
        # for aircraft nr. idx (scalar)
        if torta < -90. : # -999 signals there is no RTA defined in remainder of route
            return False

        deltime = torta-bs.sim.simt # Remaining time to next RTA [s] in simtime
        if deltime>0: # Still possible?
            gsrta = calcvrta(bs.traf.gs[idx], xtorta,
                             deltime, bs.traf.perf.axmax[idx])

            # Subtract tail wind speed vector
            tailwind = (bs.traf.windnorth[idx]*bs.traf.gsnorth[idx] + bs.traf.windeast[idx]*bs.traf.gseast[idx]) / \
                        bs.traf.gs[idx]

            # Convert to CAS
            rtacas = tas2cas(gsrta-tailwind,bs.traf.alt[idx])

            # Performance limits on speed will be applied in traf.update
            if bs.traf.actwp.spdcon[idx]<0. and bs.traf.swvnavspd[idx]:
                bs.traf.actwp.spd[idx] = rtacas
                #print("setspeedforRTA: xtorta =",xtorta)

            return rtacas
        else:
            return False

    @stack.command(name='ALT')
    def selaltcmd(self, idx: 'acid', alt: 'alt', vspd: 'vspd'=None):
        """ ALT acid, alt, [vspd] 
        
            Select autopilot altitude command."""
        bs.traf.selalt[idx]   = alt
        bs.traf.swvnav[idx]   = False

        # Check for optional VS argument
        if vspd:
            bs.traf.selvs[idx] = vspd
        else:
            if not isinstance(idx, Collection):
                idx = np.array([idx])
            delalt        = alt - bs.traf.alt[idx]
            # Check for VS with opposite sign => use default vs
            # by setting autopilot vs to zero
            oppositevs = np.logical_and(bs.traf.selvs[idx] * delalt < 0., abs(bs.traf.selvs[idx]) > 0.01)

            bs.traf.selvs[idx[oppositevs]] = 0.

    @stack.command(name='VS')
    def selvspdcmd(self, idx: 'acid', vspd:'vspd'):
        """ VS acid,vspd (ft/min)

            Vertical speed command (autopilot) """
        bs.traf.selvs[idx] = vspd #[fpm]
        # bs.traf.vs[idx] = vspd
        bs.traf.swvnav[idx] = False

    @stack.command(name='HDG', aliases=("HEADING", "TURN"))
    def selhdgcmd(self, idx: 'acid', hdg: 'hdg'):  # HDG command
        """ HDG acid,hdg (deg,True or Magnetic)
        
            Autopilot select heading command. """
        if not isinstance(idx, Collection):
            idx = np.array([idx])
        if not isinstance(hdg, Collection):
            hdg = np.array([hdg])
        # If there is wind, compute the corresponding track angle
        if bs.traf.wind.winddim > 0:
            ab50 = bs.traf.alt[idx] > 50.0 * ft
            bel50 = np.logical_not(ab50)
            iab = idx[ab50]
            ibel = idx[bel50]

            tasnorth = bs.traf.tas[iab] * np.cos(np.radians(hdg[ab50]))
            taseast = bs.traf.tas[iab] * np.sin(np.radians(hdg[ab50]))
            vnwnd, vewnd = bs.traf.wind.getdata(bs.traf.lat[iab], bs.traf.lon[iab], bs.traf.alt[iab])
            gsnorth = tasnorth + vnwnd
            gseast = taseast + vewnd
            self.trk[iab] = np.degrees(np.arctan2(gseast, gsnorth))%360.
            self.trk[ibel] = hdg
        else:
            self.trk[idx] = hdg

        bs.traf.swlnav[idx] = False
        # Everything went ok!
        return True

    @stack.command(name='SPD', aliases=("SPEED",))
    def selspdcmd(self, idx: 'acid', casmach: 'spd'):  # SPD command
        """ SPD acid, casmach (= CASkts/Mach) 
        
            Select autopilot speed. """
        # Depending on or position relative to crossover altitude,
        # we will maintain CAS or Mach when altitude changes
        # We will convert values when needed
        bs.traf.selspd[idx] = casmach

        # Used to be: Switch off VNAV: SPD command overrides
        bs.traf.swvnavspd[idx]   = False
        return True

    @stack.command(name='DEST')
    def setdest(self, acidx: 'acid', wpname:'wpt' = None):
        ''' DEST acid, latlon/airport

            Set destination of aircraft, aircraft wil fly to this airport. '''
        if wpname is None:
            return True, 'DEST ' + bs.traf.id[acidx] + ': ' + self.dest[acidx]
        route = self.route[acidx]
        apidx = bs.navdb.getaptidx(wpname)
        if apidx < 0:
            if bs.traf.ap.route[acidx].nwp > 0:
                reflat = bs.traf.ap.route[acidx].wplat[-1]
                reflon = bs.traf.ap.route[acidx].wplon[-1]
            else:
                reflat = bs.traf.lat[acidx]
                reflon = bs.traf.lon[acidx]

            success, posobj = txt2pos(wpname, reflat, reflon)
            if success:
                lat = posobj.lat
                lon = posobj.lon
            else:
                return False, "DEST: Position " + wpname + " not found."

        else:
            lat = bs.navdb.aptlat[apidx]
            lon = bs.navdb.aptlon[apidx]

        self.dest[acidx] = wpname
        iwp = route.addwpt(acidx, self.dest[acidx], route.dest,
                           lat, lon, 0.0, bs.traf.cas[acidx])
        # If only waypoint: activate
        if (iwp == 0) or (self.orig[acidx] != "" and route.nwp == 2):
            bs.traf.actwp.lat[acidx] = route.wplat[iwp]
            bs.traf.actwp.lon[acidx] = route.wplon[iwp]
            bs.traf.actwp.nextaltco[acidx] = route.wpalt[iwp]
            bs.traf.actwp.spd[acidx] = route.wpspd[iwp]

            bs.traf.swlnav[acidx] = True
            bs.traf.swvnav[acidx] = True
            route.iactwp = iwp
            route.direct(acidx, route.wpname[iwp])

        # If not found, say so
        elif iwp < 0:
            return False, ('DEST position'+self.dest[acidx] + " not found.")

    @stack.command(name='ORIG')
    def setorig(self, acidx: 'acid', wpname: 'wpt' = None):
        ''' ORIG acid, latlon/airport

            Set origin of aircraft. '''
        if wpname is None:
            return True, 'ORIG ' + bs.traf.id[acidx] + ': ' + self.orig[acidx]
        route = self.route[acidx]
        apidx = bs.navdb.getaptidx(wpname)
        if apidx < 0:
            if bs.traf.ap.route[acidx].nwp > 0:
                reflat = bs.traf.ap.route[acidx].wplat[-1]
                reflon = bs.traf.ap.route[acidx].wplon[-1]
            else:
                reflat = bs.traf.lat[acidx]
                reflon = bs.traf.lon[acidx]

            success, posobj = txt2pos(wpname, reflat, reflon)
            if success:
                lat = posobj.lat
                lon = posobj.lon
            else:
                return False, ("ORIG: Position " + wpname + " not found.")

        else:
            lat = bs.navdb.aptlat[apidx]
            lon = bs.navdb.aptlon[apidx]

        # Origin: bookkeeping only for now, store in route as origin
        self.orig[acidx] = wpname
        iwp = route.addwpt(acidx, self.orig[acidx], route.orig,
                           lat, lon, 0.0, bs.traf.cas[acidx])
        if iwp < 0:
            return False, (self.orig[acidx] + " not found.")

    @stack.command(name='VNAV')
    def setVNAV(self, idx: 'acid', flag: 'bool'=None):
        """ VNAV acid,[ON/OFF]
        
            Switch on/off VNAV mode, the vertical FMS mode (autopilot) """
        if not isinstance(idx, Collection):
            if idx is None:
                # All aircraft are targeted
                bs.traf.swvnav    = np.array(bs.traf.ntraf * [flag])
                bs.traf.swvnavspd = np.array(bs.traf.ntraf * [flag])
            else:
                # Prepare for the loop
                idx = np.array([idx])

        # Set VNAV for all aircraft in idx array
        output = []
        for i in idx:
            if flag is None:
                msg = bs.traf.id[i] + ": VNAV is " + "ON" if bs.traf.swvnav[i] else "OFF"
                if not bs.traf.swvnavspd[i]:
                    msg += " but VNAVSPD is OFF"
                output.append(bs.traf.id[i] + ": VNAV is " + "ON" if bs.traf.swvnav[i] else "OFF")

            elif flag:
                if not bs.traf.swlnav[i]:
                    return False, (bs.traf.id[i] + ": VNAV ON requires LNAV to be ON")

                route = self.route[i]
                if route.nwp > 0:
                    bs.traf.swvnav[i]    = True
                    bs.traf.swvnavspd[i] = True
                    self.route[i].calcfp()
                    actwpidx = self.route[i].iactwp
                    self.ComputeVNAV(i,self.route[i].wptoalt[actwpidx],self.route[i].wpxtoalt[actwpidx],\
                                     self.route[i].wptorta[actwpidx],self.route[i].wpxtorta[actwpidx])
                    bs.traf.actwp.nextaltco[i] = self.route[i].wptoalt[actwpidx]

                else:
                    return False, ("VNAV " + bs.traf.id[i] + ": no waypoints or destination specified")
            else:
                bs.traf.swvnav[i]    = False
                bs.traf.swvnavspd[i] = False
        if flag == None:
            return True, '\n'.join(output)

    @stack.command(name='LNAV')
    def setLNAV(self, idx: 'acid', flag: 'bool' = None):
        """ LNAV acid,[ON/OFF]

            LNAV (lateral FMS mode) switch for autopilot """
        if not isinstance(idx, Collection):
            if idx is None:
                # All aircraft are targeted
                bs.traf.swlnav = np.array(bs.traf.ntraf * [flag])
            else:
                # Prepare for the loop
                idx = np.array([idx])

        # Set LNAV for all aircraft in idx array
        output = []
        for i in idx:
            if flag is None:
                output.append(bs.traf.id[i] + ": LNAV is " + ("ON" if bs.traf.swlnav[i] else "OFF"))

            elif flag:
                route = self.route[i]
                if route.nwp <= 0:
                    return False, ("LNAV " + bs.traf.id[i] + ": no waypoints or destination specified")
                elif not bs.traf.swlnav[i]:
                    bs.traf.swlnav[i] = True
                    route.direct(i, route.wpname[route.findact(i)])
            else:
                bs.traf.swlnav[i] = False
        if flag is None:
            return True, '\n'.join(output)


    @stack.command(name='SWTOC')
    def setswtoc(self, idx: 'acid', flag: 'bool' = None):
        """ SWTOC acid,[ON/OFF]

            Switch ToC logic (=climb early) on/off"""

        if not isinstance(idx, Collection):
            if idx is None:
                # All aircraft are targeted
                self.swtoc = np.array(bs.traf.ntraf * [flag])
            else:
                # Prepare for the loop
                idx = np.array([idx])

        # Set SWTOC for all aircraft in idx array
        output = []
        for i in idx:
            if flag is None:
                output.append(bs.traf.id[i] + ": SWTOC is " + ("ON" if self.swtoc[i] else "OFF"))

            elif flag:
                self.swtoc[i] = True
            else:
                self.swtoc[i] = False
        if flag is None:
            return True, '\n'.join(output)

    @stack.command(name='SWTOD')
    def setswtod(self, idx: 'acid', flag: 'bool' = None):
        """ SWTOD acid,[ON/OFF]

            Switch ToD logic (=climb early) on/off"""
        if not isinstance(idx, Collection):
            if idx is None:
                # All aircraft are targeted
                self.swtod = np.array(bs.traf.ntraf * [flag])
            else:
                # Prepare for the loop
                idx = np.array([idx])

        # Set SWTOD for all aircraft in idx array
        output = []
        for i in idx:
            if flag is None:
                output.append(bs.traf.id[i] + ": SWTOD is " + ("ON" if self.swtoc[i] else "OFF"))

            elif flag:
                self.swtod[i] = True
            else:
                self.swtod[i] = False
        if flag is None:
            return True, '\n'.join(output)


def calcvrta(v0, dx, deltime, trafax):
    # Calculate required target ground speed v1 [m/s]
    # to meet an RTA at this leg
    #
    # Arguments are scalar
    #
    #   v0      = current ground speed [m/s]
    #   dx      = leg distance [m]
    #   deltime = time left till RTA[s]
    #   trafax  = horizontal acceleration [m/s2]

    # Set up variables
    dt = deltime

    # Do we need decelerate or accelerate
    if v0 * dt < dx:
        ax = max(0.01,abs(trafax))
    else:
        ax = -max(0.01,abs(trafax))

    # Solve 2nd order equation for v1 which results from:
    #
    #   dx = 0.5*(v0+v1)*dtacc + v1 * dtconst
    #   dt = trta - tnow = dtacc + dtconst
    #   dtacc = (v1-v0)/ax
    #
    # with unknown dtconst, dtacc, v1
    #
    # -.5/ax * v1**2  +(v0/ax+dt)*v1 -0.5*v0**2 / ax - dx =0

    a = -0.5 / ax
    b = (v0 / ax + dt)
    c = -0.5 * v0 * v0 / ax - dx

    D = b * b - 4. * a * c

    # Possibly two v1 solutions
    vlst = []

    if D >= 0.:
        x1 = (-b - sqrt(D)) / (2. * a)
        x2 = (-b + sqrt(D)) / (2. * a)

        # Check solutions for v1
        for v1 in (x1, x2):
            dtacc = (v1 - v0) / ax
            dtconst = dt - dtacc

            # Physically possible: both dtacc and dtconst >0
            if dtacc >= 0 and dtconst >= 0.:
                vlst.append(v1)

    if len(vlst) == 0:  # Not possible? Maybe borderline, so then simple calculation
        vtarg = dx/dt

    # Just in case both would be valid, take closest to v0
    elif len(vlst) == 2:
        vtarg = vlst[int(abs(vlst[1] - v0) < abs(vlst[0] - v0))]

    # Normal case is one solution
    else:
        vtarg = vlst[0]

    return vtarg

def distaccel(v0,v1,axabs):
    """Calculate distance travelled during acceleration/deceleration
    v0 = start speed, v1 = endspeed, axabs = magnitude of accel/decel
    accel/decel is detemremind by sign of v1-v0
    axabs is acceleration/deceleration of which absolute value will be used
    solve for x: x = vo*t + 1/2*a*t*t    v = v0 + a*t """
    return 0.5*np.abs(v1*v1-v0*v0)/np.maximum(.001,np.abs(axabs))
