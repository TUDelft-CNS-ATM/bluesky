""" Simple turbulence implementation."""
import numpy as np
import bluesky as bs
from bluesky.tools.aero import Rearth
from bluesky.core import Entity


class Turbulence(Entity, replaceable=True):
    """ Simple turbulence implementation."""
    def __init__(self):
        self.active = False
        self.sd = np.array([])

    def reset(self):
        self.active = False
        self.SetStandards([0, 0.1, 0.1])

    def setnoise(self, flag):
        self.active = flag

    def SetStandards(self, s):
        self.sd = np.array(s) # m/s standard turbulence  (nonnegative)
        # in (horizontal flight direction, horizontal wing direction, vertical)
        self.sd = np.where(self.sd > 1e-6, self.sd, 1e-6)

    def update(self):
        if not self.active:
            return

        timescale=np.sqrt(bs.sim.simdt)
        # Horizontal flight direction
        turbhf=np.random.normal(0,self.sd[0]*timescale,bs.traf.ntraf) #[m]

        # Horizontal wing direction
        turbhw=np.random.normal(0,self.sd[1]*timescale,bs.traf.ntraf) #[m]

        # Vertical direction
        turbalt=np.random.normal(0,self.sd[2]*timescale,bs.traf.ntraf) #[m]

        trkrad=np.radians(bs.traf.trk)
        # Lateral, longitudinal direction
        turblat=np.cos(trkrad)*turbhf-np.sin(trkrad)*turbhw #[m]
        turblon=np.sin(trkrad)*turbhf+np.cos(trkrad)*turbhw #[m]

        # Update the aircraft locations
        bs.traf.alt = bs.traf.alt + turbalt
        bs.traf.lat = bs.traf.lat + np.degrees(turblat/Rearth)
        bs.traf.lon = bs.traf.lon + np.degrees(turblon/Rearth/bs.traf.coslat)
