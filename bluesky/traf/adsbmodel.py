import numpy as np
import bluesky as bs
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

    def __init__(self):

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

        self.SetNoise(False)

    def SetNoise(self, n):
        self.transnoise = n
        self.truncated  = n
        self.transerror = [1, 100, 100 * ft]  # [degree,m,m] standard bearing, distance, altitude error
        self.trunctime  = 0  # [s]

    def create(self, n=1):
        super(ADSB, self).create(n)

        self.lastupdate[-n:] = -self.trunctime * np.random.rand(n)
        self.lat[-n:] = bs.traf.lat[-n:]
        self.lon[-n:] = bs.traf.lon[-n:]
        self.alt[-n:] = bs.traf.alt[-n:]
        self.trk[-n:] = bs.traf.trk[-n:]
        self.tas[-n:] = bs.traf.tas[-n:]
        self.gs[-n:]  = bs.traf.gs[-n:]

    def update(self, time):
        up = np.where(self.lastupdate + self.trunctime < time)
        self.lat[up] = bs.traf.lat[up]
        self.lon[up] = bs.traf.lon[up]
        self.alt[up] = bs.traf.alt[up]
        self.trk[up] = bs.traf.trk[up]
        self.tas[up] = bs.traf.tas[up]
        self.gs[up]  = bs.traf.gs[up]
        self.vs[up]  = bs.traf.vs[up]
        self.lastupdate[up] = self.lastupdate[up] + self.trunctime
