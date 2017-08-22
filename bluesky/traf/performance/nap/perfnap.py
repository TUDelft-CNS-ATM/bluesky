import numpy as np
import bluesky as bs
from bluesky.tools import aero
from bluesky.tools.trafficarrays import TrafficArrays, RegisterElementParameters
from bluesky.traf.performance.perfbase import Perf
from bluesky.traf.performance.nap import aircraft as ac
from bluesky.traf.performance.nap import phase as ph

class PerfNAP(Perf):
    """
    Open-source Nifty Aircraft Performance (NAP) Model

    Methods:
        create(): initialize new aircraft with performance parameters
        update(): update performance parameters
    """

    def __init__(self, min_update_dt=1):
        super(PerfNAP, self).__init__()

        self.min_update_dt = min_update_dt    # second, minimum update dt
        self.current_sim_time = 0       # last update simulation time
        self.ac_warning = False         # aircraft mdl to default warning
        self.eng_warning = False        # aircraft engine to default warning

        self.envelops = ac.load_all_aircraft_envelop()  # envelops for all aircraft d

        with RegisterElementParameters(self):
            self.engnum = np.array([])  # number of engines
            self.engthrust = np.array([])  # static engine thrust
            self.engbpr = np.array([]) # engine bypass ratio
            self.thrustratio = np.array([]) # thrust ratio at current alt spd
            self.fficao = np.array([])  # list of icao fuel flows from emission data bank

    def create(self):
        print("create NAP")
        pass

    def delete(self):
        pass

    def update(self):
        # update phase, infer from spd, roc, alt
        self.phase = ph.get(bs.traf.cas, bs.traf.vs, bs.traf.alt, unit='SI')

        # update limits, based on phase change
        limits = self.__construct_lim_matrix(self.actype, self.phase)
        self.vmin = limits[:, 0]
        self.vmax = limits[:, 1]
        self.vsmin = limits[:, 2]
        self.vsmax = limits[:, 3]
        self.hminalt = limits[:, 4]

        # compute thrust
        #   = number of engines x engine static thrust x thrust ratio
        self.thrustratio = ac.compute_thrust_ratio(
            self.phase, self.engbpr, bs.traf.tas, bs.traf.alt
        )
        self.thrust = self.engnum * self.engthrust * self.thrustratio

        # compute fuel flow
        self.fuelflow = ac.compute_fuel_flow(
            self.thrustratio, self.engnum, self.fficao
        )

        # update bank angle, due to phase change
        self.bank = np.where((self.phase==ph.TO) | (self.phase==ph.LD), 15, self.bank)
        self.bank = np.where((self.phase==ph.IC) | (self.phase==ph.CR) | (self.phase==ph.AP), 35, self.bank)

        return None

    def apply_limits(self, indent_v, indent_vs, indent_h):
        """ apply limits on indent speed, vertical speed, and altitude """
        allow_v = np.where(indent_v < self.vmin, self.vmin, indent_v)
        allow_v = np.where(indent_v > self.vmax, self.vmax, indent_v)

        allow_vs = np.where(indent_vs < self.vsmin, self.vsmin, indent_vs)
        allow_vs = np.where(indent_vs > self.vsmax, self.vsmax, indent_vs)

        allow_h = np.where(indent_h > self.hmaxalt, self.hmaxalt, indent_h)

        return allow_v, allow_vs, allow_h

    def __construct_lim_matrix(self, actypes, phases):
        """Compute limitations base on aircraft model and phases

        Args:
            actypes (String or 1D-array): aircraft type / model
            phases (int or 1D-array): aircraft flight phases

        Returns:
            2D-array: limitations [spd_min, spd_max. atl_min,
                alt_max, roc_min, roc_max]
        """

        nrow = len(actypes)
        ncol = 5

        # initialize the n_actype x 5 matrix
        limits = np.zeros((nrow, ncol))

        unique_ac_mdls = np.unique(actypes)

        for mdl in unique_ac_mdls:
            # for each aircraft type construct the speed, roc, and alt limits based on phase
            limits[:, 0] = np.where((actypes==mdl) & (phases==ph.NA), 0, limits[:, 0])
            limits[:, 0] = np.where((actypes==mdl) & (phases==ph.TO), self.envelops[mdl]['vminto'], limits[:, 0])
            limits[:, 0] = np.where((actypes==mdl) & (phases==ph.IC), self.envelops[mdl]['vminic'], limits[:, 0])
            limits[:, 0] = np.where((actypes==mdl) & (phases==ph.ER), self.envelops[mdl]['vminer'], limits[:, 0])
            limits[:, 0] = np.where((actypes==mdl) & (phases==ph.AP), self.envelops[mdl]['vminap'], limits[:, 0])
            limits[:, 0] = np.where((actypes==mdl) & (phases==ph.LD), self.envelops[mdl]['vminld'], limits[:, 0])
            limits[:, 0] = np.where((actypes==mdl) & (phases==ph.GD), 0, limits[:, 0])

            limits[:, 1] = np.where((actypes==mdl) & (phases==ph.NA), self.envelops[mdl]['vmaxer'], limits[:, 1])
            limits[:, 1] = np.where((actypes==mdl) & (phases==ph.TO), self.envelops[mdl]['vmaxto'], limits[:, 1])
            limits[:, 1] = np.where((actypes==mdl) & (phases==ph.IC), self.envelops[mdl]['vmaxic'], limits[:, 1])
            limits[:, 1] = np.where((actypes==mdl) & (phases==ph.ER), self.envelops[mdl]['vmaxer'], limits[:, 1])
            limits[:, 1] = np.where((actypes==mdl) & (phases==ph.AP), self.envelops[mdl]['vmaxap'], limits[:, 1])
            limits[:, 1] = np.where((actypes==mdl) & (phases==ph.LD), self.envelops[mdl]['vmaxld'], limits[:, 1])
            limits[:, 1] = np.where((actypes==mdl) & (phases==ph.GD), self.envelops[mdl]['vmaxer'], limits[:, 1])

            limits[:, 2] = np.where((actypes==mdl), self.envelops[mdl]['vsmin'], limits[:, 2])
            limits[:, 3] = np.where((actypes==mdl), self.envelops[mdl]['vsmax'], limits[:, 3])

            limits[:, 4] = np.where((actypes==mdl), self.envelops[mdl]['hmaxalt'], limits[:, 4])

        return limits

    def engchange(self, acid, engid=None):
        pass
