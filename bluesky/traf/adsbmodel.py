import numpy as np
from ..tools.aero import ft
from ..tools.dynamicarrays import DynamicArrays, RegisterElementParameters


class ADSB(DynamicArrays):
    """
    Traffic class definition    : Traffic data

    Methods:
        Traffic()            :  constructor

        create(acid,actype,aclat,aclon,achdg,acalt,acspd) : create aircraft
        delete(acid)         : delete an aircraft from traffic data
        update(sim)          : do a numerical integration step
        trafperf ()          : calculate aircraft performance parameters

    Members: see create

    Created by  : Jacco M. Hoekstra, Jerom Maas
    """

    def __init__(self, traf):

        # From here, define object arrays
        with RegisterElementParameters(self):
            # Most recent broadcast data
            self.lastupdate = np.array([])
            self.lat        = np.array([])
            self.lon        = np.array([])
            self.alt        = np.array([])
            self.trk        = np.array([])
            self.tas        = np.array([])
            self.gs         = np.array([])
            self.vs         = np.array([])

        self.traf = traf
        self.SetNoise(False)

    def SetNoise(self, n):
        self.transnoise = n
        self.truncated  = n
        self.transerror = [1, 100, 100 * ft]  # [degree,m,m] standard bearing, distance, altitude error
        self.trunctime  = 0  # [s]

    def create(self, n=1):
        super(ADSB, self).create(n)

        self.lastupdate[-n:] = -self.trunctime * np.random.rand(n)
        self.lat[-n:] = self.traf.lat[-n:]
        self.lon[-n:] = self.traf.lon[-n:]
        self.alt[-n:] = self.traf.alt[-n:]
        self.trk[-n:] = self.traf.trk[-n:]
        self.tas[-n:] = self.traf.tas[-n:]
        self.gs[-n:]  = self.traf.gs[-n:]

    def update(self, time):
        up = np.where(self.lastupdate + self.trunctime < time)
        self.lat[up] = self.traf.lat[up]
        self.lon[up] = self.traf.lon[up]
        self.alt[up] = self.traf.alt[up]
        self.trk[up] = self.traf.trk[up]
        self.tas[up] = self.traf.tas[up]
        self.gs[up]  = self.traf.gs[up]
        self.vs[up]  = self.traf.vs[up]
        self.lastupdate[up] = self.lastupdate[up] + self.trunctime
