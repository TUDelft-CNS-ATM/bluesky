import numpy as np
from ..tools.aero import ft
from ..tools.dynamicarrays import DynamicArrays

class ADSBModel(DynamicArrays):
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
        self.StartElementParameters()

        # Most recent broadcast data
        self.lastupdate= np.array([])
        self.lat    = np.array([])
        self.lon    = np.array([])
        self.alt    = np.array([])
        self.trk    = np.array([])
        self.tas    = np.array([])
        self.gs     = np.array([])
        self.vs     = np.array([])
        
        # Before here, define object arrays
        self.EndElementParameters()
        
        self.traf = traf
        self.setNoise(False)

    def setNoise(self,n):
        self.transnoise = n
        self.truncated  = n
        self.transerror = [1, 100, 100 * ft]  # [degree,m,m] standard bearing, distance, altitude error
        self.trunctime  = 5 #[s]
        
    def create(self,lat,lon,alt,trk,spd):
        self.CreateElement()
        
        self.lastupdate[-1] = -self.trunctime*np.random.rand(1)
        self.lat[-1] = lat
        self.lon[-1] = lon
        self.alt[-1] = alt
        self.trk[-1] = trk
        self.tas[-1] = spd
        self.gs[-1]  = spd
        #self.vs[-1]  = 0 this line is not necessary, as vs is created as 0
        
        
    def delete(self,idx):
        self.DeleteElement(idx)
        
    def update(self, time):
        up = np.where(self.lastupdate + self.trunctime < time)
        self.lat[up] = self.traf.lat[up]
        self.lon[up] = self.traf.lon[up]
        self.alt[up] = self.traf.alt[up]
        self.trk[up] = self.traf.trk[up]
        self.tas[up] = self.traf.tas[up]
        self.gs[up]  = self.traf.gs[up]
        self.vs[up]  = self.traf.vs[up]
        self.lastupdate[up]=self.lastupdate[up]+self.trunctime
