""" Pilot logic."""
import numpy as np
import bluesky as bs
from bluesky.core import TrafficArrays

class APorASAS(TrafficArrays):
    def __init__(self):
        super().__init__()
        with self.settrafarrays():
            # Desired aircraft states
            self.alt = np.array([])  # desired altitude [m]
            self.hdg = np.array([])  # desired heading [deg]
            self.trk = np.array([])  # desired track angle [deg]
            self.vs  = np.array([])  # desired vertical speed [m/s]
            self.tas = np.array([])  # desired speed [m/s]

    def create(self, n=1):
        super().create(n)
        self.alt[-n:] = bs.traf.alt[-n:]
        self.tas[-n:] = bs.traf.tas[-n:]
        self.hdg[-n:] = bs.traf.hdg[-n:]
        self.trk[-n:] = bs.traf.trk[-n:]

    def update(self):
        #--------- Input to Autopilot settings to follow: destination or ASAS ----------
        # Convert the ASAS commanded speed from ground speed to TAS
        if bs.traf.wind.winddim > 0:
            vwn, vwe     = bs.traf.wind.getdata(bs.traf.lat, bs.traf.lon, bs.traf.alt)
            asastasnorth = bs.traf.cr.tas * np.cos(np.radians(bs.traf.cr.trk)) - vwn
            asastaseast  = bs.traf.cr.tas * np.sin(np.radians(bs.traf.cr.trk)) - vwe
            asastas      = np.sqrt(asastasnorth**2 + asastaseast**2)
        # no wind, then ground speed = TAS
        else:
            asastas = bs.traf.cr.tas # TAS [m/s]

        # Select asas if there is a conflict AND resolution is on
        # Determine desired states per channel whether to use value from ASAS or AP.
        # bs.traf.cr.active may be used as well, will set all of these channels
        self.trk = np.where(bs.traf.cr.hdgactive, bs.traf.cr.trk, bs.traf.ap.trk)
        self.tas = np.where(bs.traf.cr.tasactive, asastas, bs.traf.ap.tas)
        self.alt = np.where(bs.traf.cr.altactive, bs.traf.cr.alt, bs.traf.ap.alt)
        self.vs  = np.where(bs.traf.cr.vsactive, bs.traf.cr.vs, bs.traf.ap.vs)

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

