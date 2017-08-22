import numpy as np
from bluesky.tools.trafficarrays import TrafficArrays, RegisterElementParameters


class PerfBase(TrafficArrays):
    def __init__(self):
        super(PerfBase, self).__init__()
        with RegisterElementParameters(self):
            # engine
            self.etype       = np.array([])  # jet, turboprop or piston
            # masses and dimensions
            self.mass        = np.array([])  # effective mass [kg]
            self.Sref        = np.array([])  # wing reference surface area [m^2]

            # flight envelope
            self.vmto        = np.array([])  # min TO spd [m/s]
            self.vmic        = np.array([])  # min. IC speed
            self.vmcr        = np.array([])  # min cruise spd
            self.vmap        = np.array([])  # min approach speed
            self.vmld        = np.array([])  # min landing spd
            self.vmin        = np.array([])  # min speed over all phases
            self.vmo         = np.array([])  # max operating speed [m/s]
            self.mmo         = np.array([])  # max operating mach number [-]

            self.hmaxact     = np.array([])  # max. altitude
            self.maxthr      = np.array([])  # maximum thrust [N]

            # Energy Share Factor
            self.ESF         = np.array([])  # [-]

            # flight phase
            self.phase       = np.array([])
            self.post_flight = np.array([])  # taxi prior of post flight?
            self.pf_flag     = np.array([])

            # performance
            self.Thr         = np.array([])  # thrust
            self.D           = np.array([])  # drag
            self.ff          = np.array([])  # fuel flow
