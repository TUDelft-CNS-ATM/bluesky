import numpy as np
from bluesky.tools.trafficarrays import TrafficArrays, RegisterElementParameters


class PerfBase(TrafficArrays):
    ''' Base class for BlueSky aircraft performance implementations. '''
    def __init__(self):
        super().__init__()

        with RegisterElementParameters(self):
            # --- fixed parameters ---
            self.actype = np.array([], dtype=str)  # aircraft type
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

    def update(self):
        """implement this method """
        pass

    def limits(self, intent_v, intent_vs, intent_h):
        """implement this method """
        pass

    def engchange(self):
        """implement this method """
        pass
