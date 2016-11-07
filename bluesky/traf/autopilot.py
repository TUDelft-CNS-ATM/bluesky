import numpy as np
from math import sin, cos, radians

from ..tools import geo
from ..tools.aero import ft, nm, vcas2tas, vtas2cas, vmach2tas, casormach
from route import Route
from ..tools.dynamicarrays import DynamicArrays, RegisterElementParameters


class Autopilot(DynamicArrays):
    def __init__(self, traf):
        self.traf = traf

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
            self.swnavvs  = np.array([])  # whether to use given VS or not
            self.swvnavvs = np.array([])  # vertical speed in VNAV

            # Traffic navigation information
            self.orig = []  # Four letter code of origin airport
            self.dest = []  # Four letter code of destination airport

        # Route objects
        self.route = []

    def create(self):
        super(Autopilot, self).create()

        # FMS directions
        self.tas[-1] = self.traf.tas[-1]
        self.trk[-1] = self.traf.trk[-1]
        self.alt[-1] = self.traf.alt[-1]
        self.spd[-1] = vtas2cas(self.tas[-1], self.alt[-1])

        # VNAV Variables
        self.dist2vs[-1] = -999.

        # Route objects
        self.route.append(Route(self.traf.navdb))

    def delete(self, idx):
        super(Autopilot, self).delete(idx)
        # Route objects
        del self.route[idx]

    def update(self, simt):
        # Scheduling: when dt has passed or restart
        if self.t0 + self.dt < simt or simt < self.t0:
            self.t0 = simt

            # FMS LNAV mode:
            qdr, dist = geo.qdrdist(self.traf.lat, self.traf.lon, self.traf.actwp.lat, self.traf.actwp.lon)  # [deg][nm])

            # Shift waypoints for aircraft i where necessary
            for i in self.traf.actwp.Reached(qdr, dist):
                # Save current wp speed
                oldspd = self.traf.actwp.spd[i]

                # Get next wp (lnavon = False if no more waypoints)
                lat, lon, alt, spd, xtoalt, toalt, lnavon, flyby, self.traf.actwp.next_qdr[i] =  \
                       self.route[i].getnextwp()  # note: xtoalt,toalt in [m]

                # End of route/no more waypoints: switch off LNAV
                self.traf.swlnav[i] = self.traf.swlnav[i] and lnavon

                # In case of no LNAV, do not allow VNAV mode on its own
                self.traf.swvnav[i] = self.traf.swvnav[i] and self.traf.swlnav[i]

                self.traf.actwp.lat[i]   = lat
                self.traf.actwp.lon[i]   = lon
                self.traf.actwp.flyby[i] = int(flyby)  # 1.0 in case of fly by, else fly over

                # User has entered an altitude for this waypoint
                if alt >= 0.:
                    self.traf.actwp.alt[i] = alt

                if spd > 0. and self.traf.swlnav[i] and self.traf.swvnav[i]:
                    # Valid speed and LNAV and VNAV ap modes are on
                    self.traf.actwp.spd[i] = spd
                else:
                    self.traf.actwp.spd[i] = -999.

                # VNAV spd mode: use speed of this waypoint as commanded speed
                # while passing waypoint and save next speed for passing next wp
                if self.traf.swvnav[i] and oldspd > 0.0:
                    dummy, self.traf.aspd[i], self.traf.ama[i] = casormach(oldspd, self.traf.alt[i])

                # VNAV = FMS ALT/SPD mode
                self.ComputeVNAV(i, toalt, xtoalt)

            #=============== End of Waypoint switching loop ===================

            # VNAV Guidance

            # Do VNAV start of descent check
            dy = (self.traf.actwp.lat - self.traf.lat)
            dx = (self.traf.actwp.lon - self.traf.lon) * self.traf.coslat
            dist2wp   = 60. * nm * np.sqrt(dx * dx + dy * dy)

            # VNAV logic: descend as late as possible, climb as soon as possible
            govertical = self.traf.swvnav * ((dist2wp < self.dist2vs)+(self.traf.actwp.alt > self.traf.alt))
            
            # If not lnav:Climb/descend if doing so before lnav/vnav was switched off
            #    (because there are no more waypoints). This is needed
            #    to continue descending when you get into a conflict
            #    while descending to the destination (the last waypoint)
            self.swnavvs = np.where(self.traf.swlnav, govertical, dist < self.traf.actwp.turndist)

            self.swvnavvs  = np.where(self.swnavvs, self.steepness * self.traf.gs, self.swvnavvs)

            self.vs = np.where(self.traf.swvnav, self.swvnavvs, self.traf.avsdef * self.traf.limvs_flag)

            self.alt = np.where(self.swnavvs, self.traf.actwp.alt, self.traf.apalt)

            # When descending or climbing in VNAV also update altitude command of select/hold mode            
            self.traf.apalt = np.where(self.swnavvs,self.traf.actwp.alt,self.traf.apalt)
            
            # LNAV commanded track angle
            self.trk = np.where(self.traf.swlnav, qdr, self.trk)

        # Below crossover altitude: CAS=const, above crossover altitude: MA = const
        self.tas = vcas2tas(self.traf.aspd, self.traf.alt) * self.traf.belco + vmach2tas(self.traf.ama, self.traf.alt) * self.traf.abco

    def ComputeVNAV(self, idx, toalt, xtoalt):
        if not (toalt >= 0 and self.traf.swvnav[idx]):
            self.dist2vs[idx] = -999
            return
#        print "alt, toalt=",self.traf.alt[idx],toalt

        # So: somewhere there is an altitude constraint ahead
        # Compute proper values for self.traf.actwp.alt, self.dist2vs, self.alt, self.traf.actwp.vs
        # Descent VNAV mode (T/D logic)
        #
        # xtoalt =  distance to go to next altitude constraint at a waypoinit in the route 
        #           (could be beyond next waypoint) 
        #        
        # toalt  = altitude at next waypoint with an altitude constraint
        #
        if self.traf.alt[idx] > toalt + 10. * ft:
            

            #Calculate max allowed altitude at next wp (above toalt)
            self.traf.actwp.alt[idx] = min(self.traf.alt[idx],toalt + xtoalt * self.steepness)
            

            # Dist to waypoint where descent should start
            self.dist2vs[idx] = (self.traf.alt[idx] - self.traf.actwp.alt[idx]) / self.steepness

            # Flat earth distance to next wp
            dy = (self.traf.actwp.lat[idx] - self.traf.lat[idx])
            dx = (self.traf.actwp.lon[idx] - self.traf.lon[idx]) * self.traf.coslat[idx]
            legdist = 60. * nm * np.sqrt(dx * dx + dy * dy)

            # If descent is urgent, descent with maximum steepness
            if legdist < self.dist2vs[idx]:
                self.alt[idx] = self.traf.actwp.alt[idx]  # dial in altitude of next waypoint as calculated

                t2go         = max(0.1, legdist) / max(0.01, self.traf.gs[idx])
                self.traf.actwp.vs[idx]  = (self.traf.actwp.alt[idx] - self.traf.alt[idx]) / t2go

            else:
                # Calculate V/S using self.steepness,
                # protect against zero/invalid ground speed value
                self.traf.actwp.vs[idx] = -self.steepness * (self.traf.gs[idx] +
                      (self.traf.gs[idx] < 0.2 * self.traf.tas[idx]) * self.traf.tas[idx])

        # Climb VNAV mode: climb as soon as possible (T/C logic)
        elif self.traf.alt[idx] < toalt - 10. * ft:
            self.traf.actwp.alt[idx] = toalt
            self.alt[idx]    = self.traf.actwp.alt[idx]  # dial in altitude of next waypoint as calculated
            self.dist2vs[idx]  = 9999.
        # Level leg: never start V/S
        else:
            self.dist2vs[idx] = -999.
                        
        return

    def selalt(self, idx, alt, vspd=None):
        """ Select altitude command: ALT acid, alt, [vspd] """
        self.traf.apalt[idx]    = alt
        self.traf.swvnav[idx]   = False

        # Check for optional VS argument
        if vspd:
            self.traf.avs[idx] = vspd
        else:
            delalt        = alt - self.traf.alt[idx]
            # Check for VS with opposite sign => use default vs
            # by setting autopilot vs to zero
            if self.traf.avs[idx] * delalt < 0. and abs(self.traf.avs[idx]) > 0.01:
                self.traf.avs[idx] = 0.

    def selvspd(self, idx, vspd):
        """ Vertical speed autopilot command: VS acid vspd """
        self.traf.avs[idx] = vspd
        # self.traf.vs[idx] = vspd
        self.traf.swvnav[idx] = False

    def selhdg(self, idx, hdg):  # HDG command
        """ Select heading command: HDG acid, hdg """

        # If there is wind, compute the corresponding track angle
        if self.traf.wind.winddim > 0:
            tasnorth = self.traf.tas[idx] * cos(radians(hdg))
            taseast  = self.traf.tas[idx] * sin(radians(hdg))
            vnwnd, vewnd = self.traf.wind.getdata(self.traf.lat[idx], self.traf.lon[idx], self.traf.alt[idx])
            gsnorth    = tasnorth + vnwnd
            gseast     = taseast  + vewnd
            trk        = np.degrees(np.arctan2(gseast, gsnorth))
        else:
            trk = hdg

        self.trk[idx]  = trk
        self.traf.swlnav[idx] = False
        # Everything went ok!
        return True

    def selspd(self, idx, casmach):  # SPD command
        """ Select speed command: SPD acid, casmach (= CASkts/Mach) """
        dummy, self.traf.aspd[idx], self.traf.ama[idx] = casormach(casmach, self.traf.alt[idx])
        # Switch off VNAV: SPD command overrides
        self.traf.swvnav[idx]   = False
        return True

    def setdestorig(self, cmd, idx, *args):
        if len(args) == 0:
            if cmd == 'DEST':
                return True, ('DEST ' + self.traf.id[idx] + ': ' + self.dest[idx])
            else:
                return True, ('ORIG ' + self.traf.id[idx] + ': ' + self.orig[idx])

        route = self.route[idx]
        if len(args) == 1:

            name = args[0]

            apidx = self.traf.navdb.getapidx(name)
            if apidx < 0:
                return False, (cmd + ": Airport " + name + " not found.")
            lat = self.traf.navdb.aplat[apidx]
            lon = self.traf.navdb.aplon[apidx]
        else:
            lat, lon = args
            name = self.traf.id[idx] + "DEST"

        if cmd == "DEST":
            self.dest[idx] = name
            iwp = route.addwpt(self.traf, idx, self.dest[idx], route.dest,
                               lat, lon, 0.0, self.traf.cas[idx])
            # If only waypoint: activate
            if (iwp == 0) or (self.orig[idx] != "" and route.nwp == 2):
                self.traf.actwp.lat[idx] = route.wplat[iwp]
                self.traf.actwp.lon[idx] = route.wplon[iwp]
                self.traf.actwp.alt[idx] = route.wpalt[iwp]
                self.traf.actwp.spd[idx] = route.wpspd[iwp]

                self.traf.swlnav[idx] = True
                self.traf.swvnav[idx] = True
                route.iactwp = iwp
                route.direct(self.traf, idx, route.wpname[iwp])

            # If not found, say so
            elif iwp < 0:
                return False, (self.dest[idx] + " not found.")

        # Origin: bookkeeping only for now
        else:
            self.orig[idx] = name
            iwp = route.addwpt(self.traf, idx, self.orig[idx], route.orig,
                               self.traf.lat[idx], self.traf.lon[idx], 0.0, self.traf.cas[idx])
            if iwp < 0:
                return False, (self.orig[idx] + " not found.")

    def setLNAV(self, idx, flag=None):
        """ Set LNAV on or off for a specific or for all aircraft """
        if idx is None:
            # All aircraft are targeted
            self.traf.swlnav = np.array(self.traf.ntraf * [flag])

        elif flag is None:
            return True, (self.traf.id[idx] + ": LNAV is " + "ON" if self.traf.swlnav[idx] else "OFF")

        elif flag:
            route = self.route[idx]
            if route.nwp <= 0:
                return False, ("LNAV " + self.traf.id[idx] + ": no waypoints or destination specified")
            elif not self.traf.swlnav[idx]:
               self.traf.swlnav[idx] = True
               route.direct(self.traf, idx, route.wpname[route.findact(self.traf, idx)])
        else:
            self.traf.swlnav[idx] = False

    def setVNAV(self, idx, flag=None):
        """ Set VNAV on or off for a specific or for all aircraft """
        if idx is None:
            # All aircraft are targeted
            self.traf.swvnav = np.array(self.traf.ntraf * [flag])

        elif flag is None:
            return True, (self.traf.id[idx] + ": VNAV is " + "ON" if self.traf.swvnav[idx] else "OFF")

        elif flag:
            if not self.traf.swlnav[idx]:
                return False, (self.traf.id[idx] + ": VNAV ON requires LNAV to be ON")

            route = self.route[idx]
            if route.nwp > 0:
                self.traf.swvnav[idx] = True
                self.route[idx].calcfp()
                self.ComputeVNAV(idx,self.route[idx].wptoalt[self.route[idx].iactwp],
                                     self.route[idx].wpxtoalt[self.route[idx].iactwp])
            else:
                return False, ("VNAV " + self.traf.id[idx] + ": no waypoints or destination specified")
        else:
            self.traf.swvnav[idx] = False

    def reset(self):
        super(Autopilot,self).reset()
        self.route = []
        
        