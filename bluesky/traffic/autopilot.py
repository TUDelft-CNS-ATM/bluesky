""" Autopilot Implementation."""
from math import sin, cos, radians, sqrt, atan
import numpy as np
try:
    from collections.abc import Collection
except ImportError:
    # In python <3.3 collections.abc doesn't exist
    from collections import Collection
import bluesky as bs
from bluesky.tools import geo
from bluesky.tools.misc import degto180
from bluesky.tools.position import txt2pos
from bluesky.tools.aero import ft, nm, vcasormach2tas, vcas2tas, tas2cas, cas2tas, g0
from bluesky.core import Entity, timed_function
from .route import Route

bs.settings.set_variable_defaults(fms_dt=10.5)


class Autopilot(Entity, replaceable=True):
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
            self.dist2vs  = np.array([])  # distance from coming waypoint to TOD
            self.swvnavvs = np.array([])  # whether to use given VS or not
            self.vnavvs   = np.array([])  # vertical speed in VNAV

            # LNAV variables
            self.qdr2wp   = np.array([]) # Direction to waypoint from the last time passing was checked
                                         # to avoid 180 turns due to updated qdr shortly before passing wp

             # Traffic navigation information
            self.orig = []  # Four letter code of origin airport
            self.dest = []  # Four letter code of destination airport

            # Route objects
            self.route = []

    def create(self, n=1):
        super().create(n)

        # FMS directions
        self.tas[-n:] = bs.traf.tas[-n:]
        self.trk[-n:] = bs.traf.trk[-n:]
        self.alt[-n:] = bs.traf.alt[-n:]

        # LNAV variables
        self.qdr2wp[-n:] = -999   # Direction to waypoint from the last time passing was checked
        # to avoid 180 turns due to updated qdr shortly before passing wp

        # VNAV Variables
        self.dist2vs[-n:] = -999.

        # Route objects
        for ridx, acid in enumerate(bs.traf.id[-n:]):
            self.route[ridx - n] = Route(acid)

    #no longer timed @timed_function(name='fms', dt=bs.settings.fms_dt, manual=True)
    def update_fms(self, qdr, dist):
        # Check which aircraft i have reached the ir active waypoint
        # Shift waypoints for aircraft i where necessary
        # Reached function return list of indices where reached logic is True
        for i in bs.traf.actwp.Reached(qdr, dist, bs.traf.actwp.flyby,
                                       bs.traf.actwp.flyturn,bs.traf.actwp.turnrad):
            # Save current wp speed for use on next leg when we pass this waypoint
            # VNAV speeds are always FROM-speed, so we accelerate/decellerate at the waypoint
            # where this speed is specified, so we need to save it for use now
            # before getting the new data for the next waypoint

            # Get speed for next leg from the waypoint we pass now
            bs.traf.actwp.spd[i]    = bs.traf.actwp.nextspd[i]
            bs.traf.actwp.spdcon[i] = bs.traf.actwp.nextspd[i]

            # Execute stack commands for the still active waypoint, which we pass
            self.route[i].runactwpstack()

            # Use turnradius of passing wp for bank angle
            if bs.traf.actwp.flyturn[i] and bs.traf.actwp.turnrad[i]>0.:
                if bs.traf.actwp.turnspd[i]>0.:
                    turnspd = bs.traf.actwp.turnspd[i]
                else:
                    turnspd = bs.traf.tas[i]

                bs.traf.aphi[i] = atan(turnspd*turnspd/(bs.traf.actwp.turnrad[i]*nm*g0)) # [rad]
            else:
                bs.traf.aphi[i] = 0.0  #[rad] or leave untouched???

            # Get next wp (lnavon = False if no more waypoints)
            lat, lon, alt, bs.traf.actwp.nextspd[i], bs.traf.actwp.xtoalt[i], toalt, \
                bs.traf.actwp.xtorta[i], bs.traf.actwp.torta[i], \
                lnavon, flyby, flyturn, turnrad, turnspd,\
                bs.traf.actwp.next_qdr[i] =      \
                self.route[i].getnextwp()  # note: xtoalt,toalt in [m]

            # End of route/no more waypoints: switch off LNAV using the lnavon
            # switch returned by getnextwp
            if not lnavon and bs.traf.swlnav[i]:
                bs.traf.swlnav[i] = False
                # Last wp: copy last wp values for alt and speed in autopilot
                if bs.traf.swvnavspd[i] and bs.traf.actwp.nextspd[i]>= 0.0:
                    bs.traf.selspd[i] = bs.traf.actwp.nextspd[i]

            # In case of no LNAV, do not allow VNAV mode on its own
            bs.traf.swvnav[i] = bs.traf.swvnav[i] and bs.traf.swlnav[i]

            bs.traf.actwp.lat[i] = lat  # [deg]
            bs.traf.actwp.lon[i] = lon  # [deg]
            # 1.0 in case of fly by, else fly over
            bs.traf.actwp.flyby[i] = int(flyby)

            # User has entered an altitude for this waypoint
            if alt >= -0.01:
                bs.traf.actwp.nextaltco[i] = alt  # [m]

            if not bs.traf.swlnav[i]:
                bs.traf.actwp.spd[i] = -999.

            # VNAV spd mode: use speed of this waypoint as commanded speed
            # while passing waypoint and save next speed for passing next wp
            # Speed is now from speed! Next speed is ready in wpdata
            if bs.traf.swvnavspd[i] and bs.traf.actwp.spd[i]> 0.0:
                    bs.traf.selspd[i] = bs.traf.actwp.spd[i]

            # Update qdr and turndist for this new waypoint for ComputeVNAV
            qdr[i], distnmi = geo.qdrdist(bs.traf.lat[i], bs.traf.lon[i],
                                        bs.traf.actwp.lat[i], bs.traf.actwp.lon[i])

            dist[i] = distnmi*nm

            # Update turndist so ComputeVNAV works, is there a next leg direction or not?
            if bs.traf.actwp.next_qdr[i] < -900.:
                local_next_qdr = qdr[i]
            else:
                local_next_qdr = bs.traf.actwp.next_qdr[i]

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

            # Calculate turn dist (and radius which we do not use) now for scalar variable [i]
            bs.traf.actwp.turndist[i], dummy = \
                bs.traf.actwp.calcturn(bs.traf.tas[i], bs.traf.bank[i],
                                        qdr[i], local_next_qdr,turnrad)  # update turn distance for VNAV

            # Reduce turn dist for reduced turnspd
            if bs.traf.actwp.flyturn[i] and bs.traf.actwp.turnrad[i]<0.0 and bs.traf.actwp.turnspd[i]>0.:
                turntas = cas2tas(bs.traf.actwp.turnspd[i], bs.traf.alt[i])
                bs.traf.actwp.turndist[i] = bs.traf.actwp.turndist[i]*turntas*turntas/(bs.traf.tas[i]*bs.traf.tas[i])

            # VNAV = FMS ALT/SPD mode incl. RTA
            self.ComputeVNAV(i, toalt, bs.traf.actwp.xtoalt[i], bs.traf.actwp.torta[i],
                             bs.traf.actwp.xtorta[i])

        # End of per waypoint i switching loop
        # Update qdr2wp with up-to-date qdr, now that we have checked passing wp
        self.qdr2wp = qdr%360.

        # Continuous guidance when speed constraint on active leg is in update-method

        # If still an RTA in the route and currently no speed constraint
        for iac in np.where((bs.traf.actwp.torta > -99.)*(bs.traf.actwp.spdcon<0.0))[0]:
            iwp = bs.traf.ap.route[iac].iactwp
            if bs.traf.ap.route[iac].wprta[iwp]>-99.:

                 # For all a/c flying to an RTA waypoint, recalculate speed more often
                dist2go4rta = geo.kwikdist(bs.traf.lat[iac],bs.traf.lon[iac], \
                                           bs.traf.actwp.lat[iac],bs.traf.actwp.lon[iac])*nm \
                               + bs.traf.ap.route[iac].wpxtorta[iwp] # last term zero for active wp rta

                # Set bs.traf.actwp.spd to rta speed, if necessary
                self.setspeedforRTA(iac,bs.traf.actwp.torta[iac],dist2go4rta)

                # If VNAV speed is on (by default coupled to VNAV), use it for speed guidance
                if bs.traf.swvnavspd[iac]:
                     bs.traf.selspd[iac] = bs.traf.actwp.spd[iac]

    def update(self):
        # FMS LNAV mode:
        # qdr[deg],distinnm[nm]
        qdr, distinnm = geo.qdrdist(bs.traf.lat, bs.traf.lon,
                                    bs.traf.actwp.lat, bs.traf.actwp.lon)  # [deg][nm])
        self.qdr2wp  = qdr
        dist2wp = distinnm*nm  # Conversion to meters

        # FMS route update and possibly waypoint shift. Note: qdr, dist2wp will be updated accordingly in case of wp switch
        self.update_fms(qdr, dist2wp) # Updates self.qdr2wp when necessary

        #================= Continuous FMS guidance ========================

        # Waypoint switching in the loop above was scalar (per a/c with index i)
        # Code below is vectorized, with arrays for all aircraft

        # Do VNAV start of descent check
        #dy = (bs.traf.actwp.lat - bs.traf.lat)  #[deg lat = 60 nm]
        #dx = (bs.traf.actwp.lon - bs.traf.lon) * bs.traf.coslat #[corrected deg lon = 60 nm]
        #dist2wp   = 60. * nm * np.sqrt(dx * dx + dy * dy) # [m]

        # VNAV logic: descend as late as possible, climb as soon as possible
        startdescent = (dist2wp < self.dist2vs) + (bs.traf.actwp.nextaltco > bs.traf.alt)

        # If not lnav:Climb/descend if doing so before lnav/vnav was switched off
        #    (because there are no more waypoints). This is needed
        #    to continue descending when you get into a conflict
        #    while descending to the destination (the last waypoint)
        #    Use 0.1 nm (185.2 m) circle in case turndist might be zero
        self.swvnavvs = bs.traf.swvnav * np.where(bs.traf.swlnav, startdescent,
                                        dist2wp <= np.maximum(185.2,bs.traf.actwp.turndist))

        #Recalculate V/S based on current altitude and distance to next alt constraint
        # How much time do we have before we need to descend?

        t2go2alt = np.maximum(0.,(dist2wp + bs.traf.actwp.xtoalt - bs.traf.actwp.turndist)) \
                                    / np.maximum(0.5,bs.traf.gs)

        # use steepness to calculate V/S unless we need to descend faster
        bs.traf.actwp.vs = np.maximum(self.steepness*bs.traf.gs, \
                                np.abs((bs.traf.actwp.nextaltco-bs.traf.alt))  \
                                /np.maximum(1.0,t2go2alt))

        self.vnavvs  = np.where(self.swvnavvs, bs.traf.actwp.vs, self.vnavvs)
        #was: self.vnavvs  = np.where(self.swvnavvs, self.steepness * bs.traf.gs, self.vnavvs)

        # self.vs = np.where(self.swvnavvs, self.vnavvs, bs.traf.apvsdef * bs.traf.limvs_flag)
        selvs    = np.where(abs(bs.traf.selvs) > 0.1, bs.traf.selvs, bs.traf.apvsdef) # m/s
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
        turntas       = np.where(bs.traf.actwp.turnspd>0.0, vcas2tas(bs.traf.actwp.turnspd, bs.traf.alt),
                                 -1.0+0.*bs.traf.tas)
        swturnspd     = bs.traf.actwp.flyturn*(turntas>0.0)*(bs.traf.actwp.turnspd>0.0)
        turntasdiff   = np.maximum(0.,(bs.traf.tas - turntas)*(turntas>0.0))

        # dt = dv/a ;  dx = v*dt + 0.5*a*dt2 => dx = dv/a * (v + 0.5*dv)
        ax = bs.traf.perf.acceleration()
        dxturnspdchg  = np.where(swturnspd, np.abs(turntasdiff)/np.maximum(0.01,ax)*(bs.traf.tas+0.5*np.abs(turntasdiff)),
                                                                   0.0*bs.traf.tas)

        # Decelerate or accelerate for next required speed because of speed constraint or RTA speed
        nexttas   = vcasormach2tas(bs.traf.actwp.nextspd,bs.traf.alt)
        tasdiff   = (nexttas - bs.traf.tas)*(bs.traf.actwp.spd>=0.) # [m/s]

        # dt = dv/a   dx = v*dt + 0.5*a*dt2 => dx = dv/a * (v + 0.5*dv)
        dxspdconchg = np.abs(tasdiff)/np.maximum(0.01, np.abs(ax)) * (bs.traf.tas+0.5*np.abs(tasdiff))


        # Check also whether VNAVSPD is on, if not, SPD SEL has override for next leg
        # and same for turn logic
        usenextspdcon = (dist2wp < dxspdconchg)*(bs.traf.actwp.nextspd> -990.) * \
                            bs.traf.swvnavspd*bs.traf.swvnav*bs.traf.swlnav
        useturnspd = np.logical_or(bs.traf.actwp.turntonextwp,(dist2wp < dxturnspdchg+bs.traf.actwp.turndist) *\
                                                              swturnspd*bs.traf.swvnavspd*bs.traf.swvnav*bs.traf.swlnav)

        # Hold turn mode can only be switched on here, cannot be switched off here (happeps upon passing wp)
        bs.traf.actwp.turntonextwp = np.logical_or(bs.traf.actwp.turntonextwp,useturnspd)

        # Which CAS/Mach do we have to keep? VNAV, last turn or next turn?
        oncurrentleg = (abs(degto180(bs.traf.trk - qdr)) < 2.0) # [deg]
        inoldturn    = (bs.traf.actwp.oldturnspd > 0.) * np.logical_not(oncurrentleg)

        # Avoid using old turning speeds when turning of this leg to the next leg
        # by disabling (old) turningspd when on leg
        bs.traf.actwp.oldturnspd = np.where(oncurrentleg*(bs.traf.actwp.oldturnspd>0.), -999.,
                                            bs.traf.actwp.oldturnspd)

        # turnfromlastwp can only be switched off here, not on (latter happens upon passing wp)
        bs.traf.actwp.turnfromlastwp = np.logical_and(bs.traf.actwp.turnfromlastwp,inoldturn)

        # Select speed: turn sped, next speed constraint, or current speed constraint
        bs.traf.selspd = np.where(useturnspd,bs.traf.actwp.turnspd,
                                  np.where(usenextspdcon, bs.traf.actwp.nextspd,
                                           np.where((bs.traf.actwp.spdcon>0)*bs.traf.swvnavspd,bs.traf.actwp.spd,
                                                                            bs.traf.selspd)))


        # Temporary override when still in old turn
        bs.traf.selspd = np.where(inoldturn*(bs.traf.actwp.oldturnspd>0.)*bs.traf.swvnavspd*bs.traf.swvnav*bs.traf.swlnav,
                                  bs.traf.actwp.oldturnspd,bs.traf.selspd)

        #debug if inoldturn[0]:
        #debug     print("inoldturn bs.traf.trk =",bs.traf.trk[0],"qdr =",qdr)
        #debug elif usenextspdcon[0]:
        #debug     print("usenextspdcon")
        #debug elif useturnspd[0]:
        #debug     print("useturnspd")
        #debug elif bs.traf.actwp.spdcon>0:
        #debug     print("using current speed constraint")
        #debug else:
        #debug     print("no speed given")

        # Below crossover altitude: CAS=const, above crossover altitude: Mach = const
        self.tas = vcasormach2tas(bs.traf.selspd, bs.traf.alt)

        return

    def ComputeVNAV(self, idx, toalt, xtoalt, torta, xtorta):
        # debug print ("ComputeVNAV for",bs.traf.id[idx],":",toalt/ft,"ft  ",xtoalt/nm,"nm")

        # Check if there is a target altitude and VNAV is on, else return doing nothing
        if toalt < 0 or not bs.traf.swvnav[idx]:
            self.dist2vs[idx] = -999. #dist to next wp will never be less than this, so VNAV will do nothing
            return

        # Flat earth distance to next wp
        dy = (bs.traf.actwp.lat[idx] - bs.traf.lat[idx])  # [deg lat = 60. nm]
        dx = (bs.traf.actwp.lon[idx] - bs.traf.lon[idx]) * bs.traf.coslat[idx]  # [corrected deg lon = 60. nm]
        legdist = 60. * nm * np.sqrt(dx * dx + dy * dy)  # [m]

        # Check  whether active waypoint speed needs to be adjusted for RTA
        # sets bs.traf.actwp.spd, if necessary
        #debug print("xtorta+legdist =",(xtorta+legdist)/nm)
        self.setspeedforRTA(idx, torta, xtorta+legdist) # all scalar

        # So: somewhere there is an altitude constraint ahead
        # Compute proper values for bs.traf.actwp.nextaltco, self.dist2vs, self.alt, bs.traf.actwp.vs
        # Descent VNAV mode (T/D logic)
        #
        # xtoalt  =  distance to go to next altitude constraint at a waypoint in the route
        #           (could be beyond next waypoint) [m]
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
        # - Ignore and look beyond waypoints without an altidue constraint
        # - Climb as soon as possible after previous altitude constraint
        #   and climb as fast as possible, so arriving at alt earlier is ok
        # - Descend at the latest when necessary for next altitude constraint
        #   which can be many waypoints beyond current actual waypoint


        # VNAV Descent mode
        if bs.traf.alt[idx] > toalt + 10. * ft:


            #Calculate max allowed altitude at next wp (above toalt)
            bs.traf.actwp.nextaltco[idx] = min(bs.traf.alt[idx],toalt + xtoalt * self.steepness) # [m] next alt constraint
            bs.traf.actwp.xtoalt[idx]    = xtoalt # [m] distance to next alt constraint measured from next waypoint


            # Dist to waypoint where descent should start [m]
            self.dist2vs[idx] = bs.traf.actwp.turndist[idx] + \
                               np.abs(bs.traf.alt[idx] - bs.traf.actwp.nextaltco[idx]) / self.steepness
            #print("self.dist2vs=",self.dist2vs)

            # Flat earth distance to next wp
            dy = (bs.traf.actwp.lat[idx] - bs.traf.lat[idx])   # [deg lat = 60. nm]
            dx = (bs.traf.actwp.lon[idx] - bs.traf.lon[idx]) * bs.traf.coslat[idx] # [corrected deg lon = 60. nm]
            legdist = 60. * nm * np.sqrt(dx * dx + dy * dy)  # [m]


            # If the descent is urgent, descend with maximum steepness
            if legdist < self.dist2vs[idx]: # [m]
                self.alt[idx] = bs.traf.actwp.nextaltco[idx]  # dial in altitude of next waypoint as calculated

                t2go         = max(0.1, legdist + xtoalt) / max(0.01, bs.traf.gs[idx])
                bs.traf.actwp.vs[idx]  = (bs.traf.actwp.nextaltco[idx] - bs.traf.alt[idx]) / t2go

            else:
                # Calculate V/S using self.steepness,
                # protect against zero/invalid ground speed value
#                if bs.sim.simt>11.99:
#                    print("bs.traf.tas.shape =",bs.traf.tas.shape)
#                    print("bs.traf.gs.shape =", bs.traf.gs.shape)
#                    print("bs.traf.actwp.vs.shape =", bs.traf.actwp.vs.shape)

                bs.traf.actwp.vs[idx] = -self.steepness * (bs.traf.gs[idx] +
                      (bs.traf.gs[idx] < 0.2 * bs.traf.tas[idx]) * bs.traf.tas[idx])

        # VNAV climb mode: climb as soon as possible (T/C logic)
        elif bs.traf.alt[idx] < toalt - 10. * ft:

            # Altitude we want to climb to: next alt constraint in our route (could be further down the route)
            bs.traf.actwp.nextaltco[idx] = toalt   # [m]
            bs.traf.actwp.xtoalt[idx]    = xtoalt  # [m] distance to next alt constraint measured from next waypoint
            self.alt[idx]          = bs.traf.actwp.nextaltco[idx]  # dial in altitude of next waypoint as calculated
            self.dist2vs[idx]      = 99999.*nm #[m] Forces immediate climb as current distance to next wp will be less

            # Flat earth distance to next wp
            dy = (bs.traf.actwp.lat[idx] - bs.traf.lat[idx])
            dx = (bs.traf.actwp.lon[idx] - bs.traf.lon[idx]) * bs.traf.coslat[idx]
            legdist = 60. * nm * np.sqrt(dx * dx + dy * dy) # [m]
            t2go = max(0.1, legdist+xtoalt) / max(0.01, bs.traf.gs[idx])
            bs.traf.actwp.vs[idx]  = np.maximum(self.steepness*bs.traf.gs[idx], \
                            (bs.traf.actwp.nextaltco[idx] - bs.traf.alt[idx])/ t2go) # [m/s]
        # Level leg: never start V/S
        else:
            self.dist2vs[idx] = -999. # [m]


        return

    def setspeedforRTA(self, idx, torta, xtorta):
        #debug print("setspeedforRTA called, torta,xtorta =",torta,xtorta/nm)

        # Calculate required CAS to meet RTA
        # for aircraft nr. idx (scalar)
        if torta < -90. : # -999 signals there is no RTA defined in remainder of route
            return False

        deltime = torta-bs.sim.simt # Remaining time to next RTA [s] in simtime
        if deltime>0: # Still possible?
            trafax = abs(bs.traf.perf.acceleration()[idx])
            gsrta = calcvrta(bs.traf.gs[idx], xtorta, deltime, trafax)

            # Subtract tail wind speed vector
            tailwind = (bs.traf.windnorth[idx]*bs.traf.gsnorth[idx] + bs.traf.windeast[idx]*bs.traf.gseast[idx]) / \
                         bs.traf.gs[idx]*bs.traf.gs[idx]

            # Convert to CAS
            rtacas = tas2cas(gsrta-tailwind,bs.traf.alt[idx])

            # Performance limits on speed will be applied in traf.update
            if bs.traf.actwp.spdcon[idx]<0. and bs.traf.swvnavspd[idx]:
                bs.traf.actwp.spd[idx] = rtacas
                #print("setspeedforRTA: xtorta =",xtorta)

            return rtacas
        else:
            return False


    def selaltcmd(self, idx, alt, vspd=None):
        """ Select altitude command: ALT acid, alt, [vspd] """
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

    def selvspdcmd(self, idx, vspd):
        """ Vertical speed autopilot command: VS acid vspd """
        bs.traf.selvs[idx] = vspd #[fpm]
        # bs.traf.vs[idx] = vspd
        bs.traf.swvnav[idx] = False

    def selhdgcmd(self, idx, hdg):  # HDG command
        """ Select heading command: HDG acid, hdg """
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

    def selspdcmd(self, idx, casmach):  # SPD command
        """ Select speed command: SPD acid, casmach (= CASkts/Mach) """
        # Depending on or position relative to crossover altitude,
        # we will maintain CAS or Mach when altitude changes
        # We will convert values when needed
        bs.traf.selspd[idx] = casmach

        # Used to be: Switch off VNAV: SPD command overrides
        bs.traf.swvnavspd[idx]   = False
        return True

    def setdestorig(self, cmd, idx, *args):
        if len(args) == 0:
            if cmd == 'DEST':
                return True, 'DEST ' + bs.traf.id[idx] + ': ' + self.dest[idx]
            return True, 'ORIG ' + bs.traf.id[idx] + ': ' + self.orig[idx]

        route = self.route[idx]
        name = args[0]
        apidx = bs.navdb.getaptidx(name)

        if apidx < 0:
            if cmd =="DEST" and bs.traf.ap.route[idx].nwp>0:
                reflat = bs.traf.ap.route[idx].wplat[-1]
                reflon = bs.traf.ap.route[idx].wplon[-1]
            else:
                reflat = bs.traf.lat[idx]
                reflon = bs.traf.lon[idx]

            success, posobj = txt2pos(name, reflat, reflon)
            if success:
                lat = posobj.lat
                lon = posobj.lon
            else:
                return False, (cmd + ": Position " + name + " not found.")

        else:
            lat = bs.navdb.aptlat[apidx]
            lon = bs.navdb.aptlon[apidx]


        if cmd == "DEST":
            self.dest[idx] = name
            iwp = route.addwpt(idx, self.dest[idx], route.dest,
                               lat, lon, 0.0, bs.traf.cas[idx])
            # If only waypoint: activate
            if (iwp == 0) or (self.orig[idx] != "" and route.nwp == 2):
                bs.traf.actwp.lat[idx]       = route.wplat[iwp]
                bs.traf.actwp.lon[idx]       = route.wplon[iwp]
                bs.traf.actwp.nextaltco[idx] = route.wpalt[iwp]
                bs.traf.actwp.spd[idx]       = route.wpspd[iwp]

                bs.traf.swlnav[idx] = True
                bs.traf.swvnav[idx] = True
                route.iactwp = iwp
                route.direct(idx, route.wpname[iwp])

            # If not found, say so
            elif iwp < 0:
                return False, ('DEST'+self.dest[idx] + " not found.")

        # Origin: bookkeeping only for now, store in route as origin
        else:
            self.orig[idx] = name
            apidx = bs.navdb.getaptidx(name)

            if apidx < 0:

                if cmd =="ORIG" and bs.traf.ap.route[idx].nwp>0:
                    reflat = bs.traf.ap.route[idx].wplat[0]
                    reflon = bs.traf.ap.route[idx].wplon[0]
                else:
                    reflat = bs.traf.lat[idx]
                    reflon = bs.traf.lon[idx]

                success, posobj = txt2pos(name, reflat, reflon)
                if success:
                    lat = posobj.lat
                    lon = posobj.lon
                else:
                    return False, (cmd + ": Orig " + name + " not found.")


            iwp = route.addwpt(idx, self.orig[idx], route.orig,
                               lat, lon, 0.0, bs.traf.cas[idx])
            if iwp < 0:
                return False, (self.orig[idx] + " not found.")

    def setLNAV(self, idx, flag=None):
        """ Set LNAV on or off for specific or for all aircraft """
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
        if flag == None:
            return True, '\n'.join(output)

    def setVNAV(self, idx, flag=None):
        """ Set VNAV on or off for specific or for all aircraft """
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

