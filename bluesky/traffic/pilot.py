""" Pilot logic."""
import numpy as np
import bluesky as bs
from bluesky.tools.aero import vtas2eas, vcas2tas, vcas2mach, vtas2cas
from bluesky.tools.trafficarrays import TrafficArrays, RegisterElementParameters
from bluesky import settings


class Pilot(TrafficArrays):
    def __init__(self):
        super(Pilot, self).__init__()
        with RegisterElementParameters(self):
            # Desired aircraft states
            self.alt = np.array([])  # desired altitude [m]
            self.hdg = np.array([])  # desired heading [deg]
            self.trk = np.array([])  # desired track angle [deg]
            self.vs  = np.array([])  # desired vertical speed [m/s]
            self.tas = np.array([])  # desired speed [m/s]

    def create(self, n=1):
        super(Pilot, self).create(n)

        self.alt[-n:] = bs.traf.alt[-n:]
        self.tas[-n:] = bs.traf.tas[-n:]
        self.hdg[-n:] = bs.traf.hdg[-n:]
        self.trk[-n:] = bs.traf.trk[-n:]

    def APorASAS(self):
        #--------- Input to Autopilot settings to follow: destination or ASAS ----------
        # Convert the ASAS commanded speed from ground speed to TAS
        if bs.traf.wind.winddim > 0:
            vwn, vwe     = bs.traf.wind.getdata(bs.traf.lat, bs.traf.lon, bs.traf.alt)
            asastasnorth = bs.traf.asas.tas * np.cos(np.radians(bs.traf.asas.trk)) - vwn
            asastaseast  = bs.traf.asas.tas * np.sin(np.radians(bs.traf.asas.trk)) - vwe
            asastas      = np.sqrt(asastasnorth**2 + asastaseast**2)
        # no wind, then ground speed = TAS
        else:
            asastas = bs.traf.asas.tas # TAS [m/s]

        # Determine desired states from ASAS or AP. Select asas if there is a conflict AND resolution is on.
        self.trk = np.where(bs.traf.asas.active, bs.traf.asas.trk, bs.traf.ap.trk)
        self.tas = np.where(bs.traf.asas.active, asastas, bs.traf.ap.tas)
        self.alt = np.where(bs.traf.asas.active, bs.traf.asas.alt, bs.traf.ap.alt)
        self.vs  = np.where(bs.traf.asas.active, bs.traf.asas.vs, bs.traf.ap.vs)

        # ASAS can give positive and negative VS, but the sign of VS is determined using delalt in Traf.ComputeAirSpeed
        # Therefore, ensure that pilot.vs is always positive to prevent opposite signs of delalt and VS in Traf.ComputeAirSpeed
        self.vs = np.abs(self.vs)

        # Compute the desired heading needed to compensate for the wind
        if bs.traf.wind.winddim > 0:

            # Calculate wind correction
            vwn, vwe = bs.traf.wind.getdata(bs.traf.lat, bs.traf.lon, bs.traf.alt)
            Vw       = np.sqrt(vwn * vwn + vwe * vwe)
            winddir  = np.arctan2(vwe, vwn)
            drift    = np.radians(self.trk) - winddir  # [rad]
            steer    = np.arcsin(np.minimum(1.0, np.maximum(-1.0,
                                     Vw * np.sin(drift) / np.maximum(0.001, bs.traf.tas))))
            # desired heading
            self.hdg = (self.trk + np.degrees(steer)) % 360.
        else:
            self.hdg = self.trk % 360.

    def applylimits(self):
        # check for the flight envelope
        if settings.performance_model == 'openap':
            self.tas, self.vs, self.alt = bs.traf.perf.limits(self.tas, self.vs, self.alt, bs.traf.ax)
        else:
            bs.traf.perf.limits() # Sets limspd_flag and limspd when it needs to be limited

            # Update desired sates with values within the flight envelope
            # When CAs is limited, it needs to be converted to TAS as only this TAS is used later on!

            self.tas = np.where(bs.traf.limspd_flag, vcas2tas(bs.traf.limspd, bs.traf.alt), self.tas)

            # Autopilot selected altitude [m]
            self.alt = np.where(bs.traf.limalt_flag, bs.traf.limalt, self.alt)

            # Autopilot selected vertical speed (V/S)
            self.vs = np.where(bs.traf.limvs_flag, bs.traf.limvs, self.vs)
