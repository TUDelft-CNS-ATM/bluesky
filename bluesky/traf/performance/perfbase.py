import numpy as np
from bluesky.tools.trafficarrays import TrafficArrays, RegisterElementParameters

class PerfBase(TrafficArrays):
    def __init__(self):
        super().__init__()

        with RegisterElementParameters(self):
            # --- fixed parameters ---
            self.actype = np.array([])  # aircraft type
            self.Sref = np.array([])  # wing reference surface area [m^2]
            self.engtype = np.array([])  # integer, aircraft.ENG_TF...

            # --- dynamic parameters ---
            self.mass = np.array([])  # effective mass [kg]
            self.phase = np.array([])
            self.cd0 = np.array([])
            self.k = np.array([])
            self.bank = np.array([])
            self.thrust = np.array([])  # thrust
            self.drag = np.array([])  # drag
            self.fuelflow = np.array([])  # fuel flow

    def create(self, n):
        super().create(n)

    def delete(self, idx):
        super().delete(idx)

    def reset(self):
        super().reset()

    def update(self):
        """implement this methods"""
        pass

    def apply_limits(self):
        """implement this methods"""
        pass

    def engchange(self):
        """implement this methods"""
        pass
