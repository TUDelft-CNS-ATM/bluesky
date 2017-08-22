import numpy as np
from bluesky.tools.trafficarrays import TrafficArrays, RegisterElementParameters


class Perf(TrafficArrays):
    def __init__(self):
        super(Perf, self).__init__()

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

    def create(self):
        raise RuntimeError("function not implemented.")

    def delete(self):
        raise RuntimeError("function not implemented.")

    def thrust(self):
        raise RuntimeError("function not implemented.")

    def drag(self):
        raise RuntimeError("function not implemented.")

    def fuelflow(self):
        raise RuntimeError("function not implemented.")

    def esf(self):
        raise RuntimeError("function not implemented.")

    def bank(self):
        raise RuntimeError("function not implemented.")

    def update(self, simt):
        raise RuntimeError("function not implemented.")

    def limit(self, *args):
        raise RuntimeError("function not implemented.")

    def engchange(self):
        raise RuntimeError("function not implemented.")
