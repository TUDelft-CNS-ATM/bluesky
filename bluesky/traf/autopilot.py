""" Autopilot Implementation."""
from math import sin, cos, radians
import numpy as np
import bluesky as bs
from bluesky.tools import geo
from bluesky.tools.position import txt2pos
from bluesky.tools.aero import ft, nm, vcas2tas, vtas2cas, vmach2tas, cas2mach, \
     mach2cas, vcasormach
from .route import Route
from bluesky.tools.trafficarrays import TrafficArrays, RegisterElementParameters


class Autopilot(TrafficArrays):
    def __init__(self):
        super(Autopilot, self).__init__()
        # Scheduling of FMS and ASAS
        self.t0 = -999.  # last time fms was called
        self.dt = 1.01   # interval for fms

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
        self.spd[-n:] = vtas2cas(self.tas[-n:], self.alt[-n:])

        # VNAV Variables
        self.dist2vs[-n:] = -999.

        # Route objects
        self.route.extend([Route()] * n)

    def delete(self, idx):
        super(Autopilot, self).delete(idx)
        # Route objects
        del self.route[idx]

    def update(self, simt):
        # Scheduling: when dt has passed or restart
        if self.t0 + self.dt < simt or simt < self.t0:
            self.t0 = simt

            # FMS LNAV mode:
            qdr, dist = geo.qdrdist(bs.traf.lat, bs.traf.lon,
                                    bs.traf.actwp.lat, bs.traf.actwp.lon)  # [deg][nm])

            # Shift waypoints for aircraft i where necessary
            for i in bs.traf.actwp.Reached(qdr, dist,bs.traf.actwp.flyby):
                # Save current wp speed
                oldspd = bs.traf.actwp.spd[i]

                # Get next wp (lnavon = False if no more waypoints)
                lat, lon, alt, spd, bs.traf.actwp.xtoalt, toalt, \
                          lnavon, flyby, bs.traf.actwp.next_qdr[i] =  \
                       self.route[i].getnextwp()  # note: xtoalt,toalt in [m]

                # End of route/no more waypoints: switch off LNAV
                bs.traf.swlnav[i] = bs.traf.swlnav[i] and lnavon

                # In case of no LNAV, do not allow VNAV mode on its own
                bs.traf.swvnav[i] = bs.traf.swvnav[i] and bs.traf.swlnav[i]

                bs.traf.actwp.lat[i]   = lat
                bs.traf.actwp.lon[i]   = lon
                bs.traf.actwp.flyby[i] = int(flyby)  # 1.0 in case of fly by, else fly over

                # User has entered an altitude for this waypoint
                if alt >= 0.:
                    bs.traf.actwp.alt[i] = alt

                if spd > 0. and bs.traf.swlnav[i] and bs.traf.swvnav[i]:
                    # Valid speed and LNAV and VNAV ap modes are on
                    bs.traf.actwp.spd[i] = spd
                else:
                    bs.traf.actwp.spd[i] = -999.

                # VNAV spd mode: use speed of this waypoint as commanded speed
                # while passing waypoint and save next speed for passing next wp
                # Speed is now from speed! Next speed is ready in wpdata
                if bs.traf.swvnav[i] and oldspd > 0.0:
                    destalt = alt if alt > 0.0 else bs.traf.alt[i]
                    if oldspd<2.0:
                        bs.traf.selspd[i] = mach2cas(oldspd, destalt)
                        bs.traf.ama[i]    = oldspd
                    else:
                        bs.traf.selspd[i] = oldspd
                        bs.traf.ama[i]    = cas2mach(oldspd, destalt)

                # VNAV = FMS ALT/SPD mode
                self.ComputeVNAV(i, toalt, bs.traf.actwp.xtoalt)

            #=============== End of Waypoint switching loop ===================

            #================= Continuous FMS guidance ========================
            # Do VNAV start of descent check
            dy = (bs.traf.actwp.lat - bs.traf.lat)
            dx = (bs.traf.actwp.lon - bs.traf.lon) * bs.traf.coslat
            dist2wp   = 60. * nm * np.sqrt(dx * dx + dy * dy)

            # VNAV logic: descend as late as possible, climb as soon as possible
            startdescent = bs.traf.swvnav * ((dist2wp < self.dist2vs)+(bs.traf.actwp.alt > bs.traf.alt))

            # If not lnav:Climb/descend if doing so before lnav/vnav was switched off
            #    (because there are no more waypoints). This is needed
            #    to continue descending when you get into a conflict
            #    while descending to the destination (the last waypoint)
            #    Use 100 nm (185.2 m) circle in case turndist might be zero
            self.swvnavvs = np.where(bs.traf.swlnav, startdescent, dist <= np.maximum(185.2,bs.traf.actwp.turndist))

            #Recalculate V/S based on current altitude and distance to next alt constraint
            t2go2alt = np.maximum(0.,(dist2wp + bs.traf.actwp.xtoalt - bs.traf.actwp.turndist*nm)) \
                                        / np.maximum(0.5,bs.traf.gs)
                                        
            bs.traf.actwp.vs = np.maximum(self.steepness*bs.traf.gs, \
                                   np.abs((bs.traf.actwp.alt-bs.traf.alt))/np.maximum(1.0,t2go2alt))

            self.vnavvs  = np.where(self.swvnavvs, bs.traf.actwp.vs, self.vnavvs)
            #was: self.vnavvs  = np.where(self.swvnavvs, self.steepness * bs.traf.gs, self.vnavvs)

            # self.vs = np.where(self.swvnavvs, self.vnavvs, bs.traf.apvsdef * bs.traf.limvs_flag)
            selvs = np.where(abs(bs.traf.selvs) > 0.1, bs.traf.selvs, bs.traf.apvsdef) # m/s
            self.vs = np.where(self.swvnavvs, self.vnavvs, selvs * bs.traf.limvs_flag)

            self.alt = np.where(self.swvnavvs, bs.traf.actwp.alt, bs.traf.selalt)

            # When descending or climbing in VNAV also update altitude command of select/hold mode
            bs.traf.selalt = np.where(self.swvnavvs,bs.traf.actwp.alt,bs.traf.selalt)

            # LNAV commanded track angle
            self.trk = np.where(bs.traf.swlnav, qdr, self.trk)

            # FMS speed guidance: anticipate accel distance

            # Actual distance it takes to decelerate
            nexttas, nextcas, nextmach = vcasormach(bs.traf.actwp.spd,bs.traf.alt) 
            tasdiff  = nexttas - bs.traf.tas # [m/s]
            dtspdchg = np.abs(tasdiff)/np.maximum(0.01,np.abs(bs.traf.ax))
            dxspdchg = 0.5*np.sign(tasdiff)*np.abs(bs.traf.ax)*dtspdchg*dtspdchg + bs.traf.tas*dtspdchg
            
            spdcon         = bs.traf.actwp.spd > 0.
            bs.traf.selspd = np.where(spdcon*(dist2wp < dxspdchg)*bs.traf.swvnav, nextcas, bs.traf.selspd)
            bs.traf.ama    = np.where(spdcon*(dist2wp < dxspdchg)*bs.traf.swvnav, nextmach, bs.traf.ama)
           
        # Below crossover altitude: CAS=const, above crossover altitude: Mach = const
        self.tas = bs.traf.belco *vcas2tas(bs.traf.selspd, bs.traf.alt)  + \
                      bs.traf.abco*vmach2tas(bs.traf.ama, bs.traf.alt)


    def ComputeVNAV(self, idx, toalt, xtoalt):
        if not (toalt >= 0 and bs.traf.swvnav[idx]):
            self.dist2vs[idx] = -999
            return

        # So: somewhere there is an altitude constraint ahead
        # Compute proper values for bs.traf.actwp.alt, self.dist2vs, self.alt, bs.traf.actwp.vs
        # Descent VNAV mode (T/D logic)
        #
        # xtoalt =  distance to go to next altitude constraint at a waypoinit in the route
        #           (could be beyond next waypoint)
        #
        # toalt  = altitude at next waypoint with an altitude constraint
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
            bs.traf.actwp.alt[idx] = min(bs.traf.alt[idx],toalt + xtoalt * self.steepness)


            # Dist to waypoint where descent should start
            self.dist2vs[idx] = bs.traf.actwp.turndist[idx]*nm + \
                               (bs.traf.alt[idx] - bs.traf.actwp.alt[idx]) / self.steepness

            # Flat earth distance to next wp
            dy = (bs.traf.actwp.lat[idx] - bs.traf.lat[idx])
            dx = (bs.traf.actwp.lon[idx] - bs.traf.lon[idx]) * bs.traf.coslat[idx]
            legdist = 60. * nm * np.sqrt(dx * dx + dy * dy)


            # If the descent is urgent, descend with maximum steepness
            if legdist < self.dist2vs[idx]:
                self.alt[idx] = bs.traf.actwp.alt[idx]  # dial in altitude of next waypoint as calculated

                t2go         = max(0.1, legdist + xtoalt) / max(0.01, bs.traf.gs[idx])
                bs.traf.actwp.vs[idx]  = (bs.traf.actwp.alt[idx] - bs.traf.alt[idx]) / t2go

            else:
                # Calculate V/S using self.steepness,
                # protect against zero/invalid ground speed value
                bs.traf.actwp.vs[idx] = -self.steepness * (bs.traf.gs[idx] +
                      (bs.traf.gs[idx] < 0.2 * bs.traf.tas[idx]) * bs.traf.tas[idx])

        # VNAV climb mode: climb as soon as possible (T/C logic)
        elif bs.traf.alt[idx] < toalt - 10. * ft:


            bs.traf.actwp.alt[idx] = toalt
            self.alt[idx]          = bs.traf.actwp.alt[idx]  # dial in altitude of next waypoint as calculated
            self.dist2vs[idx]      = 9999.

            # Flat earth distance to next wp
            dy = (bs.traf.actwp.lat[idx] - bs.traf.lat[idx])
            dx = (bs.traf.actwp.lon[idx] - bs.traf.lon[idx]) * bs.traf.coslat[idx]
            legdist = 60. * nm * np.sqrt(dx * dx + dy * dy) # [m]
            t2go = max(0.1, legdist+xtoalt) / max(0.01, bs.traf.gs[idx])
            bs.traf.actwp.vs[idx]  = np.maximum(self.steepness*bs.traf.gs[idx], \
                            (bs.traf.actwp.alt[idx] - bs.traf.alt[idx])/ t2go) # [m/s]
        # Level leg: never start V/S
        else:
            self.dist2vs[idx] = -999.

        return

    def selaltcmd(self, idx, alt, vspd=None):
        """ Select altitude command: ALT acid, alt, [vspd] """
        if idx < 0 or idx >= bs.traf.ntraf:
            return False, "ALT: Aircraft does not exist"

        bs.traf.selalt[idx]    = alt
        bs.traf.swvnav[idx]   = False

        # Check for optional VS argument
        if vspd:
            bs.traf.selvs[idx] = vspd
        else:
            delalt        = alt - bs.traf.alt[idx]
            # Check for VS with opposite sign => use default vs
            # by setting autopilot vs to zero
            if bs.traf.selvs[idx] * delalt < 0. and abs(bs.traf.selvs[idx]) > 0.01:
                bs.traf.selvs[idx] = 0.

    def selvspdcmd(self, idx, vspd):
        """ Vertical speed autopilot command: VS acid vspd """
        if idx < 0 or idx >= bs.traf.ntraf:
            return False, "VS: Aircraft does not exist"

        bs.traf.selvs[idx] = vspd
        # bs.traf.vs[idx] = vspd
        bs.traf.swvnav[idx] = False

    def selhdgcmd(self, idx, hdg):  # HDG command
        """ Select heading command: HDG acid, hdg """

        if idx<0 or idx>=bs.traf.ntraf:
            return False,"HDG: Aircraft does not exist"


        # If there is wind, compute the corresponding track angle
        if bs.traf.wind.winddim > 0:
            tasnorth = bs.traf.tas[idx] * cos(radians(hdg))
            taseast  = bs.traf.tas[idx] * sin(radians(hdg))
            vnwnd, vewnd = bs.traf.wind.getdata(bs.traf.lat[idx], bs.traf.lon[idx], bs.traf.alt[idx])
            gsnorth    = tasnorth + vnwnd
            gseast     = taseast  + vewnd
            trk        = np.degrees(np.arctan2(gseast, gsnorth))
        else:
            trk = hdg

        self.trk[idx]  = trk
        bs.traf.swlnav[idx] = False
        # Everything went ok!
        return True

    def selspdcmd(self, idx, casmach):  # SPD command
        """ Select speed command: SPD acid, casmach (= CASkts/Mach) """

        if idx<0 or idx>=bs.traf.ntraf:
            return False,"SPD: Aircraft does not exist"

        # convert input speed to cas and mach depending on the magnitude of input
        if 0.0<= casmach <= 2.0:
            bs.traf.selspd[idx] = mach2cas(casmach, bs.traf.alt[idx])
            bs.traf.ama[idx]    = casmach
        else:
            bs.traf.selspd[idx] = casmach
            bs.traf.ama[idx]    = cas2mach(casmach, bs.traf.alt[idx])


        # Switch off VNAV: SPD command overrides
        bs.traf.swvnav[idx]   = False
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
                bs.traf.actwp.lat[idx] = route.wplat[iwp]
                bs.traf.actwp.lon[idx] = route.wplon[iwp]
                bs.traf.actwp.alt[idx] = route.wpalt[iwp]
                bs.traf.actwp.spd[idx] = route.wpspd[iwp]

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
        """ Set LNAV on or off for a specific or for all aircraft """
        if idx is None:
            # All aircraft are targeted
            bs.traf.swlnav = np.array(bs.traf.ntraf * [flag])

        elif flag is None:
            return True, (bs.traf.id[idx] + ": LNAV is " + "ON" if bs.traf.swlnav[idx] else "OFF")

        elif flag:
            route = self.route[idx]
            if route.nwp <= 0:
                return False, ("LNAV " + bs.traf.id[idx] + ": no waypoints or destination specified")
            elif not bs.traf.swlnav[idx]:
               bs.traf.swlnav[idx] = True
               route.direct(bs.traf, idx, route.wpname[route.findact(idx)])
        else:
            bs.traf.swlnav[idx] = False

    def setVNAV(self, idx, flag=None):
        """ Set VNAV on or off for a specific or for all aircraft """
        if idx is None:
            # All aircraft are targeted
            bs.traf.swvnav = np.array(bs.traf.ntraf * [flag])

        elif flag is None:
            return True, (bs.traf.id[idx] + ": VNAV is " + "ON" if bs.traf.swvnav[idx] else "OFF")

        elif flag:
            if not bs.traf.swlnav[idx]:
                return False, (bs.traf.id[idx] + ": VNAV ON requires LNAV to be ON")

            route = self.route[idx]
            if route.nwp > 0:
                bs.traf.swvnav[idx] = True
                self.route[idx].calcfp()
                self.ComputeVNAV(idx,self.route[idx].wptoalt[self.route[idx].iactwp],
                                     self.route[idx].wpxtoalt[self.route[idx].iactwp])
            else:
                return False, ("VNAV " + bs.traf.id[idx] + ": no waypoints or destination specified")
        else:
            bs.traf.swvnav[idx] = False

    def reset(self):
        super(Autopilot,self).reset()
        self.route = []
