""" BlueSky traffic implementation."""
from __future__ import print_function
try:
    from collections.abc import Collection
except ImportError:
    # In python <3.3 collections.abc doesn't exist
    from collections import Collection
import numpy as np
from math import *
from random import randint
import bluesky as bs
from bluesky.tools import geo
from bluesky.tools.misc import latlon2txt
from bluesky.tools.aero import fpm, kts, ft, g0, Rearth, nm, tas2cas,\
                         vatmos,  vtas2cas, vtas2mach, vcasormach
from bluesky.tools.simtime import timed_function
from bluesky.tools.trafficarrays import TrafficArrays, RegisterElementParameters

from bluesky.traffic.asas import ConflictDetection, ConflictResolution
from .windsim import WindSim
from .conditional import Condition
from .trails import Trails
from .adsbmodel import ADSB
from .pilot import Pilot
from .autopilot import Autopilot
from .activewpdata import ActiveWaypoint
from .turbulence import Turbulence
from .trafficgroups import TrafficGroups


# Register settings defaults
bs.settings.set_variable_defaults(performance_model='openap', asas_dt=1.0)

if bs.settings.performance_model == 'bada':
    try:
        print('Using BADA Performance model')
        from .performance.bada.perfbada import PerfBADA as Perf
    except Exception as err:# ImportError as err:
        print(err)
        print('Falling back to Open Aircraft Performance (OpenAP) model')
        bs.settings.performance_model = "openap"
        from .performance.openap import OpenAP as Perf
elif bs.settings.performance_model == 'openap':
    print('Using Open Aircraft Performance (OpenAP) model')
    from .performance.openap import OpenAP as Perf
else:
    print('Using BlueSky legacy performance model')
    from .performance.legacy.perfbs import PerfBS as Perf


class Traffic(TrafficArrays):
    """
    Traffic class definition    : Traffic data
    Methods:
        Traffic()            :  constructor
        reset()              :  Reset traffic database w.r.t a/c data
        create(acid,actype,aclat,aclon,achdg,acalt,acspd) : create aircraft
        delete(acid)         : delete an aircraft from traffic data
        deletall()           : delete all traffic
        update(sim)          : do a numerical integration step
        id2idx(name)         : return index in traffic database of given call sign
        engchange(i,engtype) : change engine type of an aircraft
        setnoise(A)          : Add turbulence
    Members: see create
    Created by  : Jacco M. Hoekstra
    """

    def __init__(self):
        super(Traffic, self).__init__()

        # Traffic is the toplevel trafficarrays object
        TrafficArrays.SetRoot(self)

        self.ntraf = 0

        self.cond = Condition()  # Conditional commands list
        self.wind = WindSim()
        self.turbulence = Turbulence()
        self.translvl = 5000.*ft # [m] Default transition level

        with RegisterElementParameters(self):
            # Aircraft Info
            self.id      = []  # identifier (string)
            self.type    = []  # aircaft type (string)

            # Positions
            self.lat     = np.array([])  # latitude [deg]
            self.lon     = np.array([])  # longitude [deg]
            self.distflown = np.array([])  # distance travelled [m]
            self.alt     = np.array([])  # altitude [m]
            self.hdg     = np.array([])  # traffic heading [deg]
            self.trk     = np.array([])  # track angle [deg]

            # Velocities
            self.tas     = np.array([])  # true airspeed [m/s]
            self.gs      = np.array([])  # ground speed [m/s]
            self.gsnorth = np.array([])  # ground speed [m/s]
            self.gseast  = np.array([])  # ground speed [m/s]
            self.cas     = np.array([])  # calibrated airspeed [m/s]
            self.M       = np.array([])  # mach number
            self.vs      = np.array([])  # vertical speed [m/s]

            # Atmosphere
            self.p       = np.array([])  # air pressure [N/m2]
            self.rho     = np.array([])  # air density [kg/m3]
            self.Temp    = np.array([])  # air temperature [K]
            self.dtemp   = np.array([])  # delta t for non-ISA conditions

            # Wind speeds
            self.windnorth = np.array([])  # wind speed north component a/c pos [m/s]
            self.windeast  = np.array([])  # wind speed east component a/c pos [m/s]

            # Traffic autopilot settings
            self.selspd = np.array([])  # selected spd(CAS or Mach) [m/s or -]
            self.aptas  = np.array([])  # just for initializing
            self.selalt = np.array([])  # selected alt[m]
            self.selvs  = np.array([])  # selected vertical speed [m/s]

            # Whether to perform LNAV and VNAV
            self.swlnav    = np.array([], dtype=np.bool)
            self.swvnav    = np.array([], dtype=np.bool)
            self.swvnavspd = np.array([], dtype=np.bool)

            # Flight Models
            self.cd = ConflictDetection()
            self.cr = ConflictResolution()
            self.ap = Autopilot()
            self.pilot = Pilot()
            self.adsb = ADSB()
            self.trails = Trails()
            self.actwp = ActiveWaypoint()
            self.perf = Perf()
            
            # Group Logic
            self.groups = TrafficGroups()

            # Traffic performance data
            self.apvsdef  = np.array([])  # [m/s]default vertical speed of autopilot
            self.aphi     = np.array([])  # [rad] bank angle setting of autopilot
            self.ax       = np.array([])  # [m/s2] absolute value of longitudinal accelleration
            self.bank     = np.array([])  # nominal bank angle, [radian]
            self.swhdgsel = np.array([], dtype=np.bool)  # determines whether aircraft is turning

            # limit settings
            self.limspd      = np.array([])  # limit speed
            self.limspd_flag = np.array([], dtype=np.bool)  # flag for limit spd - we have to test for max and min
            self.limalt      = np.array([])  # limit altitude
            self.limalt_flag = np.array([])  # A need to limit altitude has been detected
            self.limvs       = np.array([])  # limit vertical speed due to thrust limitation
            self.limvs_flag  = np.array([])  # A need to limit V/S detected

            # Display information on label
            self.label       = []  # Text and bitmap of traffic label

            # Miscallaneous
            self.coslat = np.array([])  # Cosine of latitude for computations
            self.eps    = np.array([])  # Small nonzero numbers

        # Default bank angles per flight phase
        self.bphase = np.deg2rad(np.array([15, 35, 35, 35, 15, 45]))

    def reset(self):
        # This ensures that the traffic arrays (which size is dynamic)
        # are all reset as well, so all lat,lon,sdp etc but also objects adsb
        super(Traffic, self).reset()
        self.ntraf = 0

        # reset performance model
        self.perf.reset()

        # Reset models
        self.wind.clear()

        # Build new modules for turbulence
        self.turbulence.reset()

        # Noise (turbulence, ADBS-transmission noise, ADSB-truncated effect)
        self.setnoise(False)

        # Reset transition level to default value
        self.translvl = 5000.*ft

    def create(self, n=1, actype="B744", acalt=None, acspd=None, dest=None,
                aclat=None, aclon=None, achdg=None, acid=None):
        """ Create multiple random aircraft in a specified area """
        area = bs.scr.getviewbounds()
        if acid is None:
            idtmp = chr(randint(65, 90)) + chr(randint(65, 90)) + '{:>05}'
            acid = [idtmp.format(i) for i in range(n)]

        elif isinstance(acid, str):
            # Check if not already exist
            if self.id.count(acid.upper()) > 0:
                return False, acid + " already exists."  # already exists do nothing
            acid = [acid]
        else:
            # TODO: for a list of a/c, check each callsign
            pass

        super(Traffic, self).create(n)

        # Increase number of aircraft
        self.ntraf += n

        if aclat is None:
            aclat = np.random.rand(n) * (area[1] - area[0]) + area[0]
        elif isinstance(aclat, (float, int)):
            aclat = np.array(n * [aclat])

        if aclon is None:
            aclon = np.random.rand(n) * (area[3] - area[2]) + area[2]
        elif isinstance(aclon, (float, int)):
            aclon = np.array(n * [aclon])

        # Limit longitude to [-180.0, 180.0]
        if n == 1:
            aclon = aclon - 360 if aclon > 180 else \
                    aclon + 360 if aclon < -180.0 else aclon
        else:
            aclon[aclon > 180.0] -= 360.0
            aclon[aclon < -180.0] += 360.0

        if achdg is None:

            achdg = np.random.randint(1, 360, n)
        elif isinstance(achdg, (float, int)):
            achdg = np.array(n * [achdg])

        if acalt is None:
            acalt = np.random.randint(2000, 39000, n) * ft
        elif isinstance(acalt, (float, int)):
            acalt = np.array(n * [acalt])

        if acspd is None:
            acspd = np.random.randint(250, 450, n) * kts
        elif isinstance(acspd,(float, int)):
            acspd = np.array(n * [acspd])

        actype = n * [actype] if isinstance(actype, str) else actype
        dest = n * [dest] if isinstance(dest, str) else dest

        # SAVEIC: save cre command when filled in
        # Special provision in case SAVEIC is on: then save individual CRE commands
        # Names of aircraft (acid) need to be recorded for saved future commands
        # And positions need to be the same in case of *MCRE"
        for i in range(n):
            bs.stack.savecmd(" ".join([ "CRE", acid[i], actype[i],
                                        str(aclat[i]), str(aclon[i]), str(int(round(achdg[i]))),
                                        str(int(round(acalt[i]/ft))),
                                        str(int(round(acspd[i]/kts)))]))

        # Aircraft Info
        self.id[-n:]   = acid
        self.type[-n:] = actype

        # Positions
        self.lat[-n:]  = aclat
        self.lon[-n:]  = aclon
        self.alt[-n:]  = acalt

        self.hdg[-n:]  = achdg
        self.trk[-n:]  = achdg

        # Velocities
        self.tas[-n:], self.cas[-n:], self.M[-n:] = vcasormach(acspd, acalt)
        self.gs[-n:]      = self.tas[-n:]
        hdgrad = np.radians(achdg)
        self.gsnorth[-n:] = self.tas[-n:] * np.cos(hdgrad)
        self.gseast[-n:] = self.tas[-n:] * np.sin(hdgrad)

        # Atmosphere
        self.p[-n:], self.rho[-n:], self.Temp[-n:] = vatmos(acalt)

        # Wind
        if self.wind.winddim > 0:
            applywind         = self.alt[-n:]> 50.*ft
            self.windnorth[-n:], self.windeast[-n:]  = self.wind.getdata(self.lat[-n:], self.lon[-n:], self.alt[-n:])
            self.gsnorth[-n:] = self.gsnorth[-n:] + self.windnorth[-n:]*applywind
            self.gseast[-n:]  = self.gseast[-n:]  + self.windeast[-n:]*applywind
            self.trk[-n:]     = np.logical_not(applywind)*achdg + \
                                applywind*np.degrees(np.arctan2(self.gseast[-n:], self.gsnorth[-n:]))
            self.gs[-n:]      = np.sqrt(self.gsnorth[-n:]**2 + self.gseast[-n:]**2)
        else:
            self.windnorth[-n:] = 0.0
            self.windeast[-n:]  = 0.0

        # Traffic performance data
        #(temporarily default values)
        self.apvsdef[-n:] = 1500. * fpm   # default vertical speed of autopilot
        self.aphi[-n:]    = np.radians(25.)  # bank angle setting of autopilot
        self.ax[-n:]      = kts           # absolute value of longitudinal accelleration
        self.bank[-n:]    = np.radians(25.)

        # Traffic autopilot settings
        self.selspd[-n:] = self.cas[-n:]
        self.aptas[-n:]  = self.tas[-n:]
        self.selalt[-n:] = self.alt[-n:]

        # Display information on label
        self.label[-n:] = n*[['', '', '', 0]]

        # Miscallaneous
        self.coslat[-n:] = np.cos(np.radians(aclat))  # Cosine of latitude for flat-earth aproximations
        self.eps[-n:] = 0.01

        # Finally call create for child TrafficArrays. This only needs to be done
        # manually in Traffic.
        self.create_children(n)

    def creconfs(self, acid, actype, targetidx, dpsi, cpa, tlosh, dH=None, tlosv=None, spd=None):
        latref  = self.lat[targetidx]  # deg
        lonref  = self.lon[targetidx]  # deg
        altref  = self.alt[targetidx]  # m
        trkref  = radians(self.trk[targetidx])
        gsref   = self.gs[targetidx]   # m/s
        vsref   = self.vs[targetidx]   # m/s
        cpa     = cpa * nm
        pzr     = bs.settings.asas_pzr * nm
        pzh     = bs.settings.asas_pzh * ft

        trk     = trkref + radians(dpsi)
        gs      = gsref if spd is None else spd
        if dH is None:
            acalt = altref
            acvs  = 0.0
        else:
            acalt = altref + dH
            tlosv = tlosh if tlosv is None else tlosv
            acvs  = vsref - np.sign(dH) * (abs(dH) - pzh) / tlosv

        # Horizontal relative velocity vector
        gsn, gse     = gs    * cos(trk),          gs    * sin(trk)
        vreln, vrele = gsref * cos(trkref) - gsn, gsref * sin(trkref) - gse
        # Relative velocity magnitude
        vrel    = sqrt(vreln * vreln + vrele * vrele)
        # Relative travel distance to closest point of approach
        drelcpa = tlosh * vrel + (0 if cpa > pzr else sqrt(pzr * pzr - cpa * cpa))
        # Initial intruder distance
        dist    = sqrt(drelcpa * drelcpa + cpa * cpa)
        # Rotation matrix diagonal and cross elements for distance vector
        rd      = drelcpa / dist
        rx      = cpa / dist
        # Rotate relative velocity vector to obtain intruder bearing
        brn     = degrees(atan2(-rx * vreln + rd * vrele,
                                 rd * vreln + rx * vrele))

        # Calculate intruder lat/lon
        aclat, aclon = geo.kwikpos(latref, lonref, brn, dist / nm)

        # convert groundspeed to CAS, and track to heading
        wn, we     = self.wind.getdata(aclat, aclon, acalt)
        tasn, tase = gsn - wn, gse - we
        acspd      = tas2cas(sqrt(tasn * tasn + tase * tase), acalt)
        achdg      = degrees(atan2(tase, tasn))

        # Create and, when necessary, set vertical speed
        self.create(1, actype, acalt, acspd, None, aclat, aclon, achdg, acid)
        self.ap.selaltcmd(len(self.lat) - 1, altref, acvs)
        self.vs[-1] = acvs

    def delete(self, idx):
        """Delete an aircraft"""
        # If this is a multiple delete, sort first for list delete
        # (which will use list in reverse order to avoid index confusion)
        if isinstance(idx, Collection):
            idx = np.sort(idx)

        # Call the actual delete function
        super(Traffic, self).delete(idx)

        # Update conditions list
        self.cond.delac(idx)

        # Update number of aircraft
        self.ntraf = len(self.lat)
        return True

    def update(self):
        # Update only if there is traffic ---------------------
        if self.ntraf == 0:
            return

        #---------- Atmosphere --------------------------------
        self.p, self.rho, self.Temp = vatmos(self.alt)

        #---------- ADSB Update -------------------------------
        self.adsb.update()

        #---------- Fly the Aircraft --------------------------
        self.ap.update()  # Autopilot logic
        self.update_asas()  # Airborne Separation Assurance
        self.pilot.APorASAS()    # Decide autopilot or ASAS

        #---------- Performance Update ------------------------
        self.perf.update()

        #---------- Limit Speeds ------------------------------
        self.pilot.applylimits()

        #---------- Kinematics --------------------------------
        self.update_airspeed()
        self.update_groundspeed()
        self.update_pos()

        #---------- Simulate Turbulence -----------------------
        self.turbulence.update()

        # Check whether new traffic state triggers conditional commands
        self.cond.update()

        #---------- Aftermath ---------------------------------
        self.trails.update()
        return

    @timed_function('asas', dt=bs.settings.asas_dt)
    def update_asas(self):
        # Conflict detection and resolution
        self.cd.update(self, self)
        self.cr.update(self.cd, self, self)

    def update_airspeed(self):
        # Compute horizontal acceleration
        delta_spd = self.pilot.tas - self.tas
        ax = self.perf.acceleration()
        need_ax = np.abs(delta_spd) > np.abs(bs.sim.simdt * ax)
        self.ax = need_ax * np.sign(delta_spd) * ax
        # Update velocities
        self.tas = np.where(need_ax, self.tas + self.ax * bs.sim.simdt, self.pilot.tas)
        self.cas = vtas2cas(self.tas, self.alt)
        self.M = vtas2mach(self.tas, self.alt)

        # Turning
        turnrate = np.degrees(g0 * np.tan(self.bank) / np.maximum(self.tas, self.eps))
        delhdg = (self.pilot.hdg - self.hdg + 180) % 360 - 180  # [deg]
        self.swhdgsel = np.abs(delhdg) > np.abs(bs.sim.simdt * turnrate)

        # Update heading
        self.hdg = np.where(self.swhdgsel, 
                            self.hdg + bs.sim.simdt * turnrate * np.sign(delhdg), self.pilot.hdg) % 360.0

        # Update vertical speed
        delta_alt = self.pilot.alt - self.alt
        self.swaltsel = np.abs(delta_alt) > np.maximum(
            10 * ft, np.abs(2 * np.abs(bs.sim.simdt * self.vs)))
        target_vs = self.swaltsel * np.sign(delta_alt) * np.abs(self.pilot.vs)
        delta_vs = target_vs - self.vs
        # print(delta_vs / fpm)
        need_az = np.abs(delta_vs) > 300 * fpm   # small threshold
        self.az = need_az * np.sign(delta_vs) * (300 * fpm)   # fixed vertical acc approx 1.6 m/s^2
        self.vs = np.where(need_az, self.vs+self.az*bs.sim.simdt, target_vs)
        self.vs = np.where(np.isfinite(self.vs), self.vs, 0)    # fix vs nan issue

    def update_groundspeed(self):
        # Compute ground speed and track from heading, airspeed and wind
        if self.wind.winddim == 0:  # no wind
            self.gsnorth  = self.tas * np.cos(np.radians(self.hdg))
            self.gseast   = self.tas * np.sin(np.radians(self.hdg))

            self.gs  = self.tas
            self.trk = self.hdg
            self.windnorth[:], self.windeast[:] = 0.0,0.0

        else:
            applywind = self.alt>50.*ft # Only apply wind when airborne

            vnwnd,vewnd = self.wind.getdata(self.lat, self.lon, self.alt)
            self.windnorth[:], self.windeast[:] = vnwnd,vewnd
            self.gsnorth  = self.tas * np.cos(np.radians(self.hdg)) + self.windnorth*applywind
            self.gseast   = self.tas * np.sin(np.radians(self.hdg)) + self.windeast*applywind

            self.gs  = np.logical_not(applywind)*self.tas + \
                       applywind*np.sqrt(self.gsnorth**2 + self.gseast**2)

            self.trk = np.logical_not(applywind)*self.hdg + \
                       applywind*np.degrees(np.arctan2(self.gseast, self.gsnorth)) % 360.

    def update_pos(self):
        # Update position
        self.alt = np.where(self.swaltsel, self.alt + self.vs * bs.sim.simdt, self.pilot.alt)
        self.lat = self.lat + np.degrees(bs.sim.simdt * self.gsnorth / Rearth)
        self.coslat = np.cos(np.deg2rad(self.lat))
        self.lon = self.lon + np.degrees(bs.sim.simdt * self.gseast / self.coslat / Rearth)
        self.distflown += self.gs * bs.sim.simdt

    def id2idx(self, acid):
        """Find index of aircraft id"""
        if not isinstance(acid, str):
            # id2idx is called for multiple id's
            # Fast way of finding indices of all ACID's in a given list
            tmp = dict((v, i) for i, v in enumerate(self.id))
            return [tmp.get(acidi, -1) for acidi in acid]
        else:
             # Catch last created id (* or # symbol)
            if acid in ('#', '*'):
                return self.ntraf - 1

            try:
                return self.id.index(acid.upper())
            except:
                return -1

    def setnoise(self, noise=None):
        """Noise (turbulence, ADBS-transmission noise, ADSB-truncated effect)"""
        if noise is None:
            return True, "Noise is currently " + ("on" if self.turbulence.active else "off")

        self.turbulence.setnoise(noise)
        self.adsb.setnoise(noise)
        return True

    def engchange(self, acid, engid):
        """Change of engines"""
        self.perf.engchange(acid, engid)
        return

    def move(self, idx, lat, lon, alt=None, hdg=None, casmach=None, vspd=None):
        self.lat[idx]      = lat
        self.lon[idx]      = lon

        if alt is not None:
            self.alt[idx]    = alt
            self.selalt[idx] = alt

        if hdg is not None:
            self.hdg[idx]  = hdg
            self.ap.trk[idx] = hdg

        if casmach is not None:
            self.tas[idx], self.selspd[idx], _ = vcasormach(casmach, alt)

        if vspd is not None:
            self.vs[idx]     = vspd
            self.swvnav[idx] = False


    def nom(self, idx):
        """ Reset acceleration back to nominal (1 kt/s^2): NOM acid """
        self.ax[idx] = kts #[m/s2]

    def poscommand(self, idxorwp):# Show info on aircraft(int) or waypoint or airport (str)
        """POS command: Show info or an aircraft, airport, waypoint or navaid"""
        # Aircraft index
        if type(idxorwp)==int and idxorwp >= 0:

            idx           = idxorwp
            acid          = self.id[idx]
            actype        = self.type[idx]
            latlon        = latlon2txt(self.lat[idx], self.lon[idx])
            alt           = round(self.alt[idx] / ft)
            hdg           = round(self.hdg[idx])
            trk           = round(self.trk[idx])
            cas           = round(self.cas[idx] / kts)
            tas           = round(self.tas[idx] / kts)
            gs            = round(self.gs[idx]/kts)
            M             = self.M[idx]
            VS            = round(self.vs[idx]/ft*60.)
            route         = self.ap.route[idx]

            # Position report
            lines = "Info on %s %s index = %d\n" %(acid, actype, idx)     \
                  + "Pos: "+latlon+ "\n"                                  \
                  + "Hdg: %03d   Trk: %03d\n"        %(hdg, trk)              \
                  + "Alt: %d ft  V/S: %d fpm\n"  %(alt,VS)                \
                  + "CAS/TAS/GS: %d/%d/%d kts   M: %.3f\n"%(cas,tas,gs,M)

            # FMS AP modes
            if self.swlnav[idx] and route.nwp > 0 and route.iactwp >= 0:

                if self.swvnav[idx]:
                    if self.swvnavspd[idx]:
                        lines = lines + "VNAV (incl.VNAVSPD), "
                    else:
                        lines = lines + "VNAV (NOT VNAVSPD), "

                lines += "LNAV to " + route.wpname[route.iactwp] + "\n"

            # Flight info: Destination and origin
            if self.ap.orig[idx] != "" or self.ap.dest[idx] != "":
                lines = lines +  "Flying"

                if self.ap.orig[idx] != "":
                    lines = lines +  " from " + self.ap.orig[idx]

                if self.ap.dest[idx] != "":
                    lines = lines +  " to " + self.ap.dest[idx]

            # Show a/c info and highlight route of aircraft in radar window
            # and pan to a/c (to show route)
            bs.scr.showroute(acid)
            return True, lines

        # Waypoint: airport, navaid or fix
        else:
            wp = idxorwp.upper()

            # Reference position for finding nearest
            reflat, reflon = bs.scr.getviewctr()

            lines = "Info on "+wp+":\n"

            # First try airports (most used and shorter, hence faster list)
            iap = bs.navdb.getaptidx(wp)
            if iap>=0:
                aptypes = ["large","medium","small"]
                lines = lines + bs.navdb.aptname[iap]+"\n"                 \
                        + "is a "+ aptypes[max(-1,bs.navdb.aptype[iap]-1)] \
                        +" airport at:\n"                                    \
                        + latlon2txt(bs.navdb.aptlat[iap],                 \
                                     bs.navdb.aptlon[iap]) + "\n"          \
                        + "Elevation: "                                      \
                        + str(int(round(bs.navdb.aptelev[iap]/ft)))        \
                        + " ft \n"

               # Show country name
                try:
                    ico = bs.navdb.cocode2.index(bs.navdb.aptco[iap].upper())
                    lines = lines + "in "+bs.navdb.coname[ico]+" ("+      \
                             bs.navdb.aptco[iap]+")"
                except:
                    ico = -1
                    lines = lines + "Country code: "+bs.navdb.aptco[iap]
                try:
                    runways = bs.navdb.rwythresholds[bs.navdb.aptid[iap]].keys()
                    if runways:
                        lines = lines + "\nRunways: " + ", ".join(runways)
                except:
                    pass

            # Not found as airport, try waypoints & navaids
            else:
                iwps = bs.navdb.getwpindices(wp,reflat,reflon)
                if iwps[0]>=0:
                    typetxt = ""
                    desctxt = ""
                    lastdesc = "XXXXXXXX"
                    for i in iwps:

                        # One line type text
                        if typetxt == "":
                            typetxt = typetxt+bs.navdb.wptype[i]
                        else:
                            typetxt = typetxt+" and "+bs.navdb.wptype[i]

                        # Description: multi-line
                        samedesc = bs.navdb.wpdesc[i]==lastdesc
                        if desctxt == "":
                            desctxt = desctxt +bs.navdb.wpdesc[i]
                            lastdesc = bs.navdb.wpdesc[i]
                        elif not samedesc:
                            desctxt = desctxt +"\n"+bs.navdb.wpdesc[i]
                            lastdesc = bs.navdb.wpdesc[i]

                        # Navaid: frequency
                        if bs.navdb.wptype[i] in ["VOR","DME","TACAN"] and not samedesc:
                            desctxt = desctxt + " "+ str(bs.navdb.wpfreq[i])+" MHz"
                        elif bs.navdb.wptype[i]=="NDB" and not samedesc:
                            desctxt = desctxt+ " " + str(bs.navdb.wpfreq[i])+" kHz"

                    iwp = iwps[0]

                    # Basic info
                    lines = lines + wp +" is a "+ typetxt       \
                           + " at\n"\
                           + latlon2txt(bs.navdb.wplat[iwp],  \
                                        bs.navdb.wplon[iwp])
                    # Navaids have description
                    if len(desctxt)>0:
                        lines = lines+ "\n" + desctxt

                    # VOR give variation
                    if bs.navdb.wptype[iwp]=="VOR":
                        lines = lines + "\nVariation: "+ \
                                     str(bs.navdb.wpvar[iwp])+" deg"


                    # How many others?
                    nother = bs.navdb.wpid.count(wp)-len(iwps)
                    if nother>0:
                        verb = ["is ","are "][min(1,max(0,nother-1))]
                        lines = lines +"\nThere "+verb + str(nother) +\
                                   " other waypoint(s) also named " + wp

                    # In which airways?
                    connect = bs.navdb.listconnections(wp, \
                                                bs.navdb.wplat[iwp],
                                                bs.navdb.wplon[iwp])
                    if len(connect)>0:
                        awset = set([])
                        for c in connect:
                            awset.add(c[0])

                        lines = lines+"\nAirways: "+"-".join(awset)


               # Try airway id
                else:  # airway
                    awid = wp
                    airway = bs.navdb.listairway(awid)
                    if len(airway)>0:
                        lines = ""
                        for segment in airway:
                            lines = lines+"Airway "+ awid + ": " + \
                                    " - ".join(segment)+"\n"
                        lines = lines[:-1] # cut off final newline
                    else:
                        return False,idxorwp+" not found as a/c, airport, navaid or waypoint"

            # Show what we found on airport and navaid/waypoint
            return True, lines

    def airwaycmd(self,key=""):
        # Show conections of a waypoint
        reflat, reflon = bs.scr.getviewctr()

        if key=="":
            return False,'AIRWAY needs waypoint or airway'

        if bs.navdb.awid.count(key)>0:
            return self.poscommand(key.upper())
        else:
            # Find connecting airway legs
            wpid = key.upper()
            iwp = bs.navdb.getwpidx(wpid,reflat,reflon)
            if iwp<0:
                return False,key + " not found."

            wplat = bs.navdb.wplat[iwp]
            wplon = bs.navdb.wplon[iwp]
            connect = bs.navdb.listconnections(key.upper(),wplat,wplon)
            if len(connect)>0:
                lines = ""
                for c in connect:
                    if len(c)>=2:
                        # Add airway, direction, waypoint
                        lines = lines+ c[0]+": to "+c[1]+"\n"
                return True, lines[:-1]  # exclude final newline
            else:
                return False,"No airway legs found for ",key

    def settrans(self,alt=-999.):
        """ Set or show transition level"""

        # in case a valid value is ginve set it
        if alt>-900.:
            if alt>0.:
                self.translvl = alt
                return True
            else:
                return False,"Transition level needs to be ft/FL and larger than zero"

        # In case no value is given, show it
        else:
            tlvl = int(round(self.translvl/ft))
            return True,"Transition level = " + \
                          str(tlvl) + "/FL" +  str(int(round(tlvl/100.)))
