import numpy as np
from math import sqrt, sin, cos

from ..tools import geo
from ..tools.aero import fpm, kts, ft, nm, g0, tas2eas, tas2mach, tas2cas, mach2tas,  \
                         mach2cas, cas2tas, cas2mach, Rearth, vatmos, \
                         vcas2tas, vtas2cas, vtas2mach, vcas2mach, vmach2tas, \
                         vcasormach, casormach
from ..tools.misc import degto180
from route import Route
from waypoint import ActiveWaypoint
from ..tools.dynamicarrays import DynamicArrays, RegisterElementParameters

class FMS(DynamicArrays):
    def __init__(self, traf):
        self.traf = traf

        # Scheduling of FMS and ASAS
        self.t0 = -999. # last time fms was called
        self.dt = 1.01  # interval for fms

        # Standard self.steepness for descent
        self.steepness = 3000. * ft / (10. * nm)
        
        # From here, define object arrays
        with RegisterElementParameters(self):
            
            # Whether to perform LNAV and VNAV
            self.lnav   = np.array([],dtype=np.bool)
            self.vnav   = np.array([],dtype=np.bool)

            # FMS directions
            self.trk = np.array([])
            self.spd = np.array([])
            self.tas = np.array([])
            self.alt = np.array([])
            self.vs  = np.array([])

            # VNAV variables
            self.TODdist = np.array([]) # distance from coming waypoint to TOD
            self.swnavvs = np.array([]) # whether to use given VS or not
            self.vnavvs = np.array([]) # vertical speed in VNAV

            # Traffic navigation information
            self.orig   = []  # Four letter code of origin airport
            self.dest   = []  # Four letter code of destination airport

            # Active Waypoint parameters
            self.WP = ActiveWaypoint(self)

        # Route objects
        self.route = []

    def create(self):
        self.CreateElement()

        # FMS directions
        self.tas[-1] = self.traf.tas[-1]
        self.trk[-1] = self.traf.trk[-1]
        self.alt[-1] = self.traf.alt[-1]
        self.spd[-1] = vtas2cas(self.tas[-1], self.alt[-1])

        # VNAV Variables
        self.TODdist[-1] = -999.

        # Active Waypoint parameters
        self.WP.create()

        # Route objects
        self.route.append(Route(self.traf.navdb))

    def delete(self,idx):
        self.DeleteElement(idx)
        # Route objects
        del self.route[idx]

    def Guide(self, simt):
        # Scheduling: when dt has passed or restart
        if self.t0 + self.dt < simt or simt < self.t0:
            self.t0 = simt

            # FMS LNAV mode:
            qdr, dist = geo.qdrdist(self.traf.lat, self.traf.lon, self.WP.lat, self.WP.lon)  # [deg][nm])

            # Shift waypoints for aircraft i where necessary
            for i in self.WP.Reached(qdr,dist):
                # Save current wp speed
                oldspd = self.WP.spd[i]

                # Get next wp (lnavon = False if no more waypoints)
                lat, lon, alt, spd, xtoalt, toalt, lnavon, flyby, self.WP.next_qdr[i] =  \
                       self.route[i].getnextwp()  # note: xtoalt,toalt in [m]

                # End of route/no more waypoints: switch off LNAV
                self.lnav[i] = lnavon

                # In case of no LNAV, do not allow VNAV mode on its own
                self.vnav[i] = self.vnav[i] and lnavon

                self.WP.lat[i]   = lat
                self.WP.lon[i]   = lon
                self.WP.flyby[i] = int(flyby)  # 1.0 in case of fly by, else fly over

                # User has entered an altitude for this waypoint
                if alt >= 0.:
                    self.WP.alt[i] = alt
                
                if spd > 0. and self.lnav[i] and self.vnav[i]:
                    # Valid speed and LNAV and VNAV ap modes are on
                    self.WP.spd[i] = spd
                else:
                    self.WP.spd[i] = -999.
                
                # VNAV spd mode: use speed of this waypoint as commanded speed
                # while passing waypoint and save next speed for passing next wp
                if self.vnav[i] and oldspd > 0.0:
                    dummy, self.traf.aspd[i], self.traf.ama[i] = casormach(oldspd,self.traf.alt[i])
                
                # VNAV = FMS ALT/SPD mode
                self.ComputeVNAV(i,toalt,xtoalt)

            #=============== End of Waypoint switching loop ===================

            # VNAV Guidance

            # Do VNAV start of descent check
            dy = (self.WP.lat - self.traf.lat)
            dx = (self.WP.lon - self.traf.lon) * self.traf.coslat
            dist2wp   = 60. * nm * np.sqrt(dx * dx + dy * dy)

            # VNAV logic: descend as late as possible, climb as soon as possible
            govertical = self.vnav*np.logical_or(dist2wp < self.TODdist, self.WP.alt > self.traf.alt)

            # If not lnav:Climb/descend if doing so before lnav/vnav was switched off
            #    (because there are no more waypoints). This is needed
                    #    to continue descending when you get into a conflict
                    #    while descending to the destination (the last waypoint)
            self.swnavvs = np.where(self.lnav, govertical, dist < self.WP.turn)

            self.vnavvs  = np.where(self.swnavvs, self.steepness*self.traf.gs, self.vnavvs)

            self.vs = np.where(self.vnav, self.vnavvs, self.traf.avsdef * self.traf.limvs_flag)
            self.alt = np.where(self.swnavvs, self.WP.alt, self.alt)
                    
            # LNAV commanded track angle
            self.trk = np.where(self.lnav, qdr, self.trk)

        # Below crossover altitude: CAS=const, above crossover altitude: MA = const
        self.tas = vcas2tas(self.spd, self.traf.alt)*self.traf.belco + vmach2tas(self.traf.ama, self.traf.alt)*self.traf.abco

    def ComputeVNAV(self, idx, toalt, xtoalt):
        if  not (toalt >=0 and self.vnav[idx]):
            self.TODdist[idx] = -999
            return
        
        # So: somewhere there is an altitude constraint ahead
        # Compute proper values for self.WP.alt, self.TODdist, self.alt, self.WP.vs
        # Descent VNAV mode (T/D logic)
        if self.traf.alt[idx] > toalt + 10. * ft:

            #Calculate max allowed altitude at next wp (above toalt)
            self.WP.alt[idx] = toalt + xtoalt * self.steepness

            # Dist to waypoint where descent should start
            self.TODdist[idx] = (self.traf.alt[idx] - self.WP.alt[idx]) / self.steepness

            # Flat earth distance to next wp
            dy = (self.WP.lat[idx] - self.traf.lat[idx])
            dx = (self.WP.lon[idx] - self.traf.lon[idx]) * self.traf.coslat[idx]
            legdist = 60. * nm * sqrt(dx * dx + dy * dy)

            # If descent is urgent, descent with maximum steepness
            if legdist < self.TODdist[idx]:
                self.alt[idx] = self.WP.alt[idx]  # dial in altitude of next waypoint as calculated

                t2go         = max(0.1, legdist) / max(0.01, self.traf.gs[idx])
                self.WP.vs[idx]  = (self.WP.alt[idx] - self.traf.alt[idx]) / t2go

            else:
                # Calculate V/s using self.steepness,
                # protect against zero/invalid ground speed value
                self.WP.vs[idx] = -self.steepness * (self.traf.gs[idx] +
                      (self.traf.gs[idx] < 0.2 * self.traf.tas[idx]) * self.traf.tas[idx])

        # Climb VNAV mode: climb as soon as possible (T/C logic)
        elif self.traf.alt[idx] < toalt - 10. * ft:
            self.WP.alt[idx] = toalt
            self.alt[idx]    = self.WP.alt[idx]  # dial in altitude of next waypoint as calculated
            self.TODdist[idx]  = 9999.

        # Level leg: never start V/S
        else:
            self.TODdist[idx] = -999.

    def selalt(self, idx, alt, vspd=None):
        """ Select altitude command: ALT acid, alt, [vspd] """
        self.traf.apalt[idx]    = alt
        self.vnav[idx]   = False

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
        self.vnav[idx] = False

    def selhdg(self, idx, hdg):  # HDG command
        """ Select heading command: HDG acid, hdg """
        
        # If there is wind, compute the corresponding track angle            
        if self.traf.wind.winddim>0:
            tasnorth = self.traf.tas[idx]*cos(radians(hdg))
            taseast  = self.traf.tas[idx]*sin(radians(hdg)) 
            vnwnd,vewnd = self.traf.wind.getdata(self.traf.lat[idx],self.traf.lon[idx],self.traf.alt[idx])
            gsnorth    = tasnorth + vnwnd
            gseast     = taseast  + vewnd
            trk        = np.degrees(np.arctan2(gseast,gsnorth))
        else:             
            trk = hdg           

        self.trk[idx]  = trk
        self.lnav[idx] = False
        # Everything went ok!
        return True

    def selspd(self, idx, casmach):  # SPD command
        """ Select speed command: SPD acid, casmach (= CASkts/Mach) """
        dummy, self.traf.aspd[-1], self.traf.ama[-1] = casormach(casmach,self.traf.alt[idx])
        # Switch off VNAV: SPD command overrides
        self.vnav[idx]   = False
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

        else:
            lat, lon = args[-2:]
            name = str(lat)+","+str(lon)
            
        if cmd == "DEST":
            self.dest[idx] = name
            iwp = route.addwpt(self.traf, idx, self.dest[idx], route.dest,
                               0.0, self.traf.cas[idx])

            # If only waypoint: activate
            if (iwp == 0) or (self.orig[idx] != "" and route.nwp == 2):
                self.WP.lat[idx] = route.wplat[iwp]
                self.WP.lon[idx] = route.wplon[iwp]
                self.WP.alt[idx] = route.wpalt[iwp]
                self.WP.spd[idx] = route.wpspd[iwp]

                self.lnav[idx] = True
                route.iactwp = iwp

            # If not found, say so
            elif iwp < 0:
                return False, (self.dest[idx] + " not found.")

        # Origin: bookkeeping only for now
        else:
            self.orig[idx] = name
            iwp = route.addwpt(self.traf, idx, self.orig[idx], route.orig,
                                0.0, self.traf.cas[idx])
            if iwp < 0:
                return False, (self.orig[idx] + " not found.")

    def setLNAV(self, idx, flag=None):
        """ Set LNAV on or off for a specific or for all aircraft """
        if idx is None:
            # All aircraft are targeted
            self.lnav = np.array(self.traf.ntraf*[flag])

        elif flag is None:
            return True, (self.traf.id[idx] + ": LNAV is " + "ON" if self.lnav[idx] else "OFF")

        elif flag:
            route = self.route[idx]
            if route.nwp > 0:
                self.lnav[idx] = True
                route.direct(self.traf, idx, route.wpname[route.findact(self.traf, idx)])
            else:
                return False, ("LNAV " + self.traf.id[idx] + ": no waypoints or destination specified")
        else:
            self.lnav[idx] = False

    def setVNAV(self, idx, flag=None):
        """ Set VNAV on or off for a specific or for all aircraft """
        if idx is None:
            # All aircraft are targeted
            self.vnav = np.array(self.traf.ntraf*[flag])

        elif flag is None:
            return True, (self.traf.id[idx] + ": VNAV is " + "ON" if self.vnav[idx] else "OFF")

        elif flag:
            if not self.lnav[idx]:
                return False, (self.traf.id[idx] + ": VNAV ON requires LNAV to be ON")

            route = self.route[idx]
            if route.nwp > 0:
                self.vnav[idx] = True
                route.direct(self.traf, idx, route.wpname[route.findact(self.traf, idx)])
            else:
                return False, ("VNAV " + self.traf.id[idx] + ": no waypoints or destination specified")
        else:
            self.vnav[idx] = False