import numpy as np
from math import *
from random import random, randint
from ..tools import datalog
from ..tools.aero import fpm, kts, ft, g0, Rearth, \
                         vatmos,  vtas2cas, vtas2mach, casormach

from ..tools.dynamicarrays import DynamicArrays, RegisterElementParameters

from windsim import WindSim

from trails import Trails
from adsbmodel import ADSB
from asas import ASAS
from pilot import Pilot
from autopilot import Autopilot
from waypoint import ActiveWaypoint
from turbulence import Turbulence
from area import Area

from .. import settings

try:
    if settings.performance_model == 'bluesky':
        from perf import Perf

    elif settings.performance_model == 'bada':
        from perfbada import PerfBADA as Perf

except ImportError as err:
    print err.args[0]
    print 'Falling back to BlueSky performance model'
    from perf import Perf


class Traffic(DynamicArrays):
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
        setNoise(A)          : Add turbulence
    Members: see create
    Created by  : Jacco M. Hoekstra
    """

    def __init__(self, navdb):
        self.wind = WindSim()

        # Define the periodic loggers
        datalog.definePeriodicLogger('SNAPLOG', 'SNAPLOG logfile.', settings.snapdt)
        datalog.definePeriodicLogger('INSTLOG', 'INSTLOG logfile.', settings.instdt)
        datalog.definePeriodicLogger('SKYLOG', 'SKYLOG logfile.', settings.skydt)

        with RegisterElementParameters(self):

            # Register the following parameters for logging
            with datalog.registerLogParameters('SNAPLOG', self):
                # Aircraft Info
                self.id      = []  # identifier (string)
                self.type    = []  # aircaft type (string)

                # Positions
                self.lat     = np.array([])  # latitude [deg]
                self.lon     = np.array([])  # longitude [deg]
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

                # Traffic autopilot settings
                self.aspd   = np.array([])  # selected spd(CAS) [m/s]
                self.aptas  = np.array([])  # just for initializing
                self.ama    = np.array([])  # selected spd above crossover altitude (Mach) [-]
                self.apalt  = np.array([])  # selected alt[m]
                self.avs    = np.array([])  # selected vertical speed [m/s]

            # Whether to perform LNAV and VNAV
            self.swlnav   = np.array([], dtype=np.bool)
            self.swvnav   = np.array([], dtype=np.bool)

            # Flight Models
            self.asas   = ASAS(self)
            self.ap     = Autopilot(self)
            self.pilot  = Pilot(self)
            self.adsb   = ADSB(self)
            self.trails = Trails(self)
            self.actwp  = ActiveWaypoint(self)

            # Traffic performance data
            self.avsdef = np.array([])  # [m/s]default vertical speed of autopilot
            self.aphi   = np.array([])  # [rad] bank angle setting of autopilot
            self.ax     = np.array([])  # [m/s2] absolute value of longitudinal accelleration
            self.bank   = np.array([])  # nominal bank angle, [radian]
            self.bphase = np.array([])  # standard bank angles per phase
            self.hdgsel = np.array([], dtype=np.bool)  # determines whether aircraft is turning

            # Crossover altitude
            self.abco   = np.array([])
            self.belco  = np.array([])

            # limit settings
            self.limspd      = np.array([])  # limit speed
            self.limspd_flag = np.array([], dtype=np.bool)  # flag for limit spd - we have to test for max and min
            self.limalt      = np.array([])  # limit altitude
            self.limvs       = np.array([])  # limit vertical speed due to thrust limitation
            self.limvs_flag  = np.array([])

            # Display information on label
            self.label       = []  # Text and bitmap of traffic label

            # Miscallaneous
            self.coslat = np.array([])  # Cosine of latitude for computations
            self.eps    = np.array([])  # Small nonzero numbers

        self.reset(navdb)

    def reset(self, navdb):
        
        # This ensures that the traffic arrays (which size is dynamic) 
        # are all reset as well, so all lat,lon,sdp etc but also objects adsb
        super(Traffic, self).reset()
        self.ntraf = 0

        # Reset models
        self.wind.clear()

        # Build new modules for area and turbulence
        self.area       = Area(self)
        self.Turbulence = Turbulence(self)

        # Noise (turbulence, ADBS-transmission noise, ADSB-truncated effect)
        self.setNoise(False)

        # Import navigation data base
        self.navdb   = navdb

        # Default: BlueSky internal performance model.
        # Insert your BADA files to the folder "BlueSky/data/coefficients/BADA"
        # for working with EUROCONTROL`s Base of Aircraft Data revision 3.12
        self.perf    = Perf(self)

    def mcreate(self, count, actype=None, alt=None, spd=None, dest=None, area=None):
        """ Create multiple random aircraft in a specified area """
        idbase = chr(randint(65, 90)) + chr(randint(65, 90))
        if actype is None:
            actype = 'B744'

        for i in xrange(count):
            acid  = idbase + '%05d' % i
            aclat = random() * (area[1] - area[0]) + area[0]
            aclon = random() * (area[3] - area[2]) + area[2]
            achdg = float(randint(1, 360))
            acalt = (randint(2000, 39000) * ft) if alt is None else alt
            acspd = (randint(250, 450) * kts) if spd is None else spd

            self.create(acid, actype, aclat, aclon, achdg, acalt, acspd)

    def create(self, acid, actype, aclat, aclon, achdg, acalt, casmach):
        """Create an aircraft"""
        # Check if not already exist
        if self.id.count(acid.upper()) > 0:
            return False, acid + " already exists."  # already exists do nothing

        super(Traffic, self).create()

        # Increase number of aircraft
        self.ntraf = self.ntraf + 1

        # Aircraft Info
        self.id[-1]   = acid.upper()
        self.type[-1] = actype

        # Positions
        self.lat[-1]  = aclat
        self.lon[-1]  = aclon
        self.alt[-1]  = acalt

        self.hdg[-1]  = achdg
        self.trk[-1]  = achdg

        # Velocities
        self.tas[-1], self.cas[-1], self.M[-1] = casormach(casmach, acalt)
        self.gs[-1]      = self.tas[-1]
        self.gsnorth[-1] = self.tas[-1] * cos(radians(self.hdg[-1]))
        self.gseast[-1]  = self.tas[-1] * sin(radians(self.hdg[-1]))

        # Atmosphere
        self.Temp[-1], self.rho[-1], self.p[-1] = vatmos(acalt)

        # Wind
        if self.wind.winddim > 0:
            vnwnd, vewnd     = self.wind.getdata(self.lat[-1], self.lon[-1], self.alt[-1])
            self.gsnorth[-1] = self.gsnorth[-1] + vnwnd
            self.gseast[-1]  = self.gseast[-1]  + vewnd
            self.trk[-1]     = np.degrees(np.arctan2(self.gseast[-1], self.gsnorth[-1]))
            self.gs[-1]      = np.sqrt(self.gsnorth[-1]**2 + self.gseast[-1]**2)

        # Traffic performance data
        #(temporarily default values)
        self.avsdef[-1] = 1500. * fpm   # default vertical speed of autopilot
        self.aphi[-1]   = radians(25.)  # bank angle setting of autopilot
        self.ax[-1]     = kts           # absolute value of longitudinal accelleration
        self.bank[-1]   = radians(25.)

        # Crossover altitude
        self.abco[-1]   = 0  # not necessary to overwrite 0 to 0, but leave for clarity
        self.belco[-1]  = 1

        # Traffic autopilot settings
        self.aspd[-1]  = self.cas[-1]
        self.aptas[-1] = self.tas[-1]
        self.apalt[-1] = self.alt[-1]

        # Display information on label
        self.label[-1] = ['', '', '', 0]

        # Miscallaneous
        self.coslat[-1] = cos(radians(aclat))  # Cosine of latitude for flat-earth aproximations
        self.eps[-1] = 0.01

        # ----- Submodules of Traffic -----
        self.ap.create()
        self.actwp.create()
        self.pilot.create()
        self.adsb.create()
        self.area.create()
        self.asas.create()
        self.perf.create()
        self.trails.create()

        #
        if self.ntraf < 2:
            self.bphase = np.deg2rad(np.array([15, 35, 35, 35, 15, 45]))

        return True

    def delete(self, acid):
        """Delete an aircraft"""

        # Look up index of aircraft
        idx = self.id2idx(acid)
        # Do nothing if not found
        if idx < 0:
            return False
        # Decrease number of aircraft
        self.ntraf = self.ntraf - 1

        # Delete all aircraft parameters
        super(Traffic, self).delete(idx)

        # ----- Submodules of Traffic -----
        self.perf.delete(idx)
        self.area.delete(idx)
        return True

    def update(self, simt, simdt):
        # Update only if there is traffic ---------------------
        if self.ntraf == 0:
            return

        #---------- Atmosphere --------------------------------
        self.p, self.rho, self.Temp = vatmos(self.alt)

        #---------- ADSB Update -------------------------------
        self.adsb.update(simt)

        #---------- Fly the Aircraft --------------------------
        self.ap.update(simt)
        self.asas.update(simt)
        self.pilot.FMSOrAsas()

        #---------- Limit Speeds ------------------------------
        self.pilot.FlightEnvelope()

        #---------- Kinematics --------------------------------
        self.ComputeAirSpeed(simdt, simt)
        self.ComputeGroundSpeed(simdt)
        self.ComputePosition(simdt)

        #---------- Performance Update ------------------------
        self.perf.perf(simt)

        #---------- Simulate Turbulence -----------------------
        self.Turbulence.Woosh(simdt)

        #---------- Aftermath ---------------------------------
        self.trails.update(simt)
        self.area.check(simt)
        return

    def ComputeAirSpeed(self, simdt, simt):
        # Acceleration
        self.delspd = self.pilot.spd - self.tas
        swspdsel = np.abs(self.delspd) > 0.4  # <1 kts = 0.514444 m/s
        ax = self.perf.acceleration(simdt)

        # Update velocities
        self.tas = self.tas + swspdsel * ax * np.sign(self.delspd) * simdt
        self.cas = vtas2cas(self.tas, self.alt)
        self.M   = vtas2mach(self.tas, self.alt)

        # Turning
        turnrate = np.degrees(g0 * np.tan(self.bank) / np.maximum(self.tas, self.eps))
        delhdg   = (self.pilot.hdg - self.hdg + 180.) % 360 - 180.  # [deg]
        self.hdgsel = np.abs(delhdg) > np.abs(2. * simdt * turnrate)

        # Update heading
        self.hdg = (self.hdg + simdt * turnrate * self.hdgsel * np.sign(delhdg)) % 360.

        # Update vertical speed
        delalt   = self.pilot.alt - self.alt
        self.swaltsel = np.abs(delalt) > np.maximum(10 * ft, np.abs(2 * simdt * np.abs(self.vs)))
        self.vs  = self.swaltsel * np.sign(delalt) * self.pilot.vs

    def ComputeGroundSpeed(self, simdt):
        # Compute ground speed and track from heading, airspeed and wind
        if self.wind.winddim == 0:  # no wind
            self.gsnorth  = self.tas * np.cos(np.radians(self.hdg))
            self.gseast   = self.tas * np.sin(np.radians(self.hdg))

            self.gs  = self.tas
            self.trk = self.hdg

        else:
            windnorth, windeast = self.wind.getdata(self.lat, self.lon, self.alt)
            self.gsnorth  = self.tas * np.cos(np.radians(self.hdg)) + windnorth
            self.gseast   = self.tas * np.sin(np.radians(self.hdg)) + windeast

            self.gs  = np.sqrt(self.gsnorth**2 + self.gseast**2)
            self.trk = np.degrees(np.arctan2(self.gseast, self.gsnorth)) % 360.

    def ComputePosition(self, simdt):
        # Update position
        self.alt = np.where(self.swaltsel, self.alt + self.vs * simdt, self.pilot.alt)
        self.lat = self.lat + np.degrees(simdt * self.gsnorth / Rearth)
        self.coslat = np.cos(np.deg2rad(self.lat))
        self.lon = self.lon + np.degrees(simdt * self.gseast / self.coslat / Rearth)

    def id2idx(self, acid):
        """Find index of aircraft id"""
        try:
            return self.id.index(acid.upper())
        except:
            return -1

    def setNoise(self, noise=None):
        """Noise (turbulence, ADBS-transmission noise, ADSB-truncated effect)"""
        if noise is None:
            return True, "Noise is currently " + ("on" if self.Turbulence.active else "off")

        self.Turbulence.SetNoise(noise)
        self.adsb.SetNoise(noise)
        return True

    def engchange(self, acid, engid):
        """Change of engines"""
        self.perf.engchange(acid, engid)
        return

    def move(self, idx, lat, lon, alt=None, hdg=None, casmach=None, vspd=None):
        self.lat[idx]      = lat
        self.lon[idx]      = lon

        if alt:
            self.alt[idx]   = alt
            self.apalt[idx] = alt

        if hdg:
            self.hdg[idx]  = hdg
            self.ap.trk[idx] = hdg

        if casmach:
            self.tas[idx], self.aspd[-1], dummy = casormach(casmach, alt)

        if vspd:
            self.vs[idx]       = vspd
            self.swvnav[idx] = False

    def nom(self, idx):
        """ Reset acceleration back to nominal (1 kt/s^2): NOM acid """
        self.ax[idx] = kts

    def acinfo(self, acid):
        idx           = self.id.index(acid)
        actype        = self.type[idx]
        lat, lon      = self.lat[idx], self.lon[idx]
        alt, hdg, trk = self.alt[idx] / ft, self.hdg[idx], self.trk[idx]
        cas           = self.cas[idx] / kts
        tas           = self.tas[idx] / kts
        route         = self.ap.route[idx]
        line = "Info on %s %s index = %d\n" % (acid, actype, idx) \
             + "Pos = %.2f, %.2f. Spd: %d kts CAS, %d kts TAS\n" % (lat, lon, cas, tas) \
             + "Alt = %d ft, Hdg = %d, Trk = %d\n" % (alt, hdg, trk)
        if self.swlnav[idx] and route.nwp > 0 and route.iactwp >= 0:
            if self.swvnav[idx]:
                line += "VNAV, "
            line += "LNAV to " + route.wpname[route.iactwp] + "\n"
        if self.ap.orig[idx] != "" or self.ap.dest[idx] != "":
            line += "Flying"
            if self.ap.orig[idx] != "":
                line += " from " + self.ap.orig[idx]
            if self.ap.dest[idx] != "":
                line += " to " + self.ap.dest[idx]

        return line
