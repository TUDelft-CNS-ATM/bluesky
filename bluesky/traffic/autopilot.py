""" Autopilot Implementation."""
from math import sin, cos, radians
import numpy as np
try:
    from collections.abc import Collection
except ImportError:
    # In python <3.3 collections.abc doesn't exist
    from collections import Collection
import bluesky as bs
from bluesky.tools import geo
from bluesky.tools.simtime import timed_function
from bluesky.tools.position import txt2pos
from bluesky.tools.aero import ft, nm, vtas2cas, cas2mach, \
     mach2cas, vcasormach2tas, vcasormach
from .route import Route
from bluesky.tools.trafficarrays import TrafficArrays, RegisterElementParameters

bs.settings.set_variable_defaults(fms_dt=1.0)

class Autopilot(TrafficArrays):
    def __init__(self):
        super(Autopilot, self).__init__()

        # Standard self.steepness for descent
        self.steepness = 3000. * ft / (10. * nm)

        # From here, define object arrays
        with RegisterElementParameters(self):

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

            # Traffic navigation information
            self.orig = []  # Four letter code of origin airport
            self.dest = []  # Four letter code of destination airport

            # Route objects
            self.route = []

    def create(self, n=1):
        super(Autopilot, self).create(n)

        # FMS directions
        self.tas[-n:] = bs.traf.tas[-n:]
        self.trk[-n:] = bs.traf.trk[-n:]
        self.alt[-n:] = bs.traf.alt[-n:]

        # VNAV Variables
        self.dist2vs[-n:] = -999.

        # Route objects
        self.route[-n:] = [Route() for _ in range(n)]

    @timed_function('fms', dt=bs.settings.fms_dt)
    def update_fms(self, qdr, dist, dt=bs.settings.fms_dt):
        # Shift waypoints for aircraft i where necessary
        for i in bs.traf.actwp.Reached(qdr, dist, bs.traf.actwp.flyby):
            # Save current wp speed for use on next leg when we pass this waypoint
            # VNAV speeds are always FROM-speed, so we accelerate/decellerate at the waypoint
            # where this speed is specified, so we need to save it for use now
            # before getting the new data for the next waypoint

            # Save speed as specified for the waypoint we pass
            oldspd = bs.traf.actwp.spd[i]

            # Get next wp (lnavon = False if no more waypoints)
            lat, lon, alt, spd, bs.traf.actwp.xtoalt[i], toalt, xtorta, torta,\
                lnavon, flyby, bs.traf.actwp.next_qdr[i] =  \
                self.route[i].getnextwp()  # note: xtoalt,toalt in [m]

            # End of route/no more waypoints: switch off LNAV
            bs.traf.swlnav[i] = bs.traf.swlnav[i] and lnavon

            # In case of no LNAV, do not allow VNAV mode on its own
            bs.traf.swvnav[i] = bs.traf.swvnav[i] and bs.traf.swlnav[i]

            bs.traf.actwp.lat[i] = lat  # [deg]
            bs.traf.actwp.lon[i] = lon  # [deg]
            # 1.0 in case of fly by, else fly over
            bs.traf.actwp.flyby[i] = int(flyby)

            # User has entered an altitude for this waypoint
            if alt >= -0.01:
                bs.traf.actwp.nextaltco[i] = alt  # [m]

            if spd > -990. and bs.traf.swlnav[i] and bs.traf.swvnav[i]:
                bs.traf.actwp.spd[i] = spd
            else:
                bs.traf.actwp.spd[i] = -999.

            # VNAV spd mode: use speed of this waypoint as commanded speed
            # while passing waypoint and save next speed for passing next wp
            # Speed is now from speed! Next speed is ready in wpdata
            if bs.traf.swvnav[i] and oldspd > 0.0:
                    bs.traf.selspd[i] = oldspd

            # Update qdr and turndist for this new waypoint for ComputeVNAV
            qdr[i], dummy = geo.qdrdist(bs.traf.lat[i], bs.traf.lon[i],
                                        bs.traf.actwp.lat[i], bs.traf.actwp.lon[i])

            # Update turndist so ComputeVNAV works, is there a next leg direction or not?
            if bs.traf.actwp.next_qdr[i] < -900.:
                local_next_qdr = qdr[i]
            else:
                local_next_qdr = bs.traf.actwp.next_qdr[i]

            # Calculate turn dist 9and radius which we do not use) now for scalar variable [i]
            bs.traf.actwp.turndist[i], dummy = \
                bs.traf.actwp.calcturn(bs.traf.tas[i], bs.traf.bank[i],
                                        qdr[i], local_next_qdr)  # update turn distance for VNAV

            # Check  whether active waypoint speed needs to be adjusted for RTA
            self.setspeedforRTA(i,torta,xtorta)

            # VNAV = FMS ALT/SPD mode
            self.ComputeVNAV(i, toalt, bs.traf.actwp.xtoalt[i])

    def update(self):
        # FMS LNAV mode:
        # qdr[deg],distinnm[nm]
        qdr, distinnm = geo.qdrdist(bs.traf.lat, bs.traf.lon,
                                    bs.traf.actwp.lat, bs.traf.actwp.lon)  # [deg][nm])
        dist = distinnm*nm  # Conversion to meters

        # FMS route update
        self.update_fms(qdr, dist)

        #================= Continuous FMS guidance ========================

        # Waypoint switching in the loop above was scalar (per a/c with index i)
        # Code below is vectorized, with arrays for all aircraft

        # Do VNAV start of descent check
        dy = (bs.traf.actwp.lat - bs.traf.lat)  #[deg lat = 60 nm]
        dx = (bs.traf.actwp.lon - bs.traf.lon) * bs.traf.coslat #[corrected deg lon = 60 nm]
        dist2wp   = 60. * nm * np.sqrt(dx * dx + dy * dy) # [m]
        #print("dist2wp =",dist2wp,"   self.dist2vs =",self.dist2vs)

        #print("actpwp.nextaltco=",bs.traf.actwp.nextaltco)

        # VNAV logic: descend as late as possible, climb as soon as possible
        startdescent = (dist2wp < self.dist2vs) + (bs.traf.actwp.nextaltco > bs.traf.alt)

        # If not lnav:Climb/descend if doing so before lnav/vnav was switched off
        #    (because there are no more waypoints). This is needed
        #    to continue descending when you get into a conflict
        #    while descending to the destination (the last waypoint)
        #    Use 0.1 nm (185.2 m) circle in case turndist might be zero
        self.swvnavvs = bs.traf.swvnav * np.where(bs.traf.swlnav, startdescent,
                                        dist <= np.maximum(185.2,bs.traf.actwp.turndist))

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
        self.trk = np.where(bs.traf.swlnav, qdr, self.trk)

        # FMS speed guidance: anticipate accel distance

        # Actual distance it takes to decelerate
        nexttas  = vcasormach2tas(bs.traf.actwp.spd,bs.traf.alt)
        tasdiff  = nexttas - bs.traf.tas # [m/s]
        dtspdchg = np.abs(tasdiff)/np.maximum(0.01,np.abs(bs.traf.ax)) #[s]
        dxspdchg = 0.5*np.sign(tasdiff)*np.abs(bs.traf.ax)*dtspdchg*dtspdchg + bs.traf.tas*dtspdchg #[m]

        # Chekc also whether VNAVSPD is on, if not, SPD SEL has override
        usespdcon      = (dist2wp < dxspdchg)*(bs.traf.actwp.spd > -990.)*bs.traf.swvnavspd*bs.traf.swvnav

        bs.traf.selspd = np.where(usespdcon, bs.traf.actwp.spd, bs.traf.selspd)

        # Below crossover altitude: CAS=const, above crossover altitude: Mach = const
        self.tas = vcasormach2tas(bs.traf.selspd, bs.traf.alt)



    def ComputeVNAV(self, idx, toalt, xtoalt):
        # debug print ("ComputeVNAV for",bs.traf.id[idx],":",toalt/ft,"ft  ",xtoalt/nm,"nm")
        # Check if there is a target altitude and VNAV is on, else return doing nothing
        if toalt < 0 or not bs.traf.swvnav[idx]:
            self.dist2vs[idx] = -999. #dist to next wp will never be less than this, so VNAV will do nothing
            return

        # So: somewhere there is an altitude constraint ahead
        # Compute proper values for bs.traf.actwp.nextaltco, self.dist2vs, self.alt, bs.traf.actwp.vs
        # Descent VNAV mode (T/D logic)
        #
        # xtoalt  =  distance to go to next altitude constraint at a waypoinit in the route
        #           (could be beyond next waypoint)
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
        # Calculate required CAS to meet RTA
        # for aircraft nr. idx (scalar)
        if torta < 0. : # no RTA defined in remainder of route
            return False

        deltime = torta-bs.sim.simt # Remaining time to next RTA [s] in simtime
        if deltime>0: # Still possible?
            # xtorta does not include speed constraint legs, and torta has been compensated for that
            gsrta = xtorta/deltime # Calculate along track ground speed

            # Subtract tail wind speed vector
            tailwind = (bs.traf.windnorth[idx]*bs.traf.gsnorth[idx] + bs.traf.windeast[idx]*bs.traf.gseast[idx]) / \
                         bs.traf.gs*bs.traf.gs

            # Convert to CAS
            rtacas = tas2cas(gsrta-tailwind,bs.traf.alt[idx])

            # Performance limits on speed will be applied in traf.update
            bs.traf.actwp.spd[idx]= rtacas

            return True
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
            self.trk[iab] = np.degrees(np.arctan2(gseast, gsnorth))
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
            else:
                return True, 'ORIG ' + bs.traf.id[idx] + ': ' + self.orig[idx]

        if idx<0 or idx>=bs.traf.ntraf:
            return False, cmd + ": Aircraft does not exist."

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
                    self.ComputeVNAV(i,self.route[i].wptoalt[actwpidx],self.route[i].wpxtoalt[actwpidx])
                    bs.traf.actwp.nextaltco[i] = self.route[i].wptoalt[actwpidx]

                else:
                    return False, ("VNAV " + bs.traf.id[i] + ": no waypoints or destination specified")
            else:
                bs.traf.swvnav[i]    = False
                bs.traf.swvnavspd[i] = False
        if flag == None:
            return True, '\n'.join(output)
