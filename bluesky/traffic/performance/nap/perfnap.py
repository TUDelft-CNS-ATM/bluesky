import numpy as np
import bluesky as bs
from bluesky.tools import aero
from bluesky.tools.trafficarrays import TrafficArrays, RegisterElementParameters
from bluesky.traffic.performance.perfbase import PerfBase
from bluesky.traffic.performance.nap import coeff, thrust
from bluesky.traffic.performance.nap import phase as ph

class PerfNAP(PerfBase):
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

        self.coeff = coeff.Coefficient()

        with RegisterElementParameters(self):
            self.lifttype = np.array([])  # lift type, fixwing [1] or rotor [2]
            self.engnum = np.array([])  # number of engines
            self.engthrust = np.array([])  # static engine thrust
            self.engbpr = np.array([])  # engine bypass ratio
            self.thrustratio = np.array([])  # thrust ratio at current alt spd
            self.ffidl = np.array([])  # icao fuel flows idle
            self.ffapp = np.array([])  # icao fuel flows appraoch
            self.ffco = np.array([])  # icao fuel flows climbout
            self.ffto = np.array([])  # icao fuel flows takeoff

    def create(self, n=1, lifttype='fixwing'):
        super(PerfNAP, self).create(n)

        if lifttype.lower() == 'fixwing':
            newactypes = []
            fficaos = []

            for actype in bs.traf.type[-n:]:
                # convert to known aircraft type
                avaliables = list(self.coeff.acs_fixwing.keys())
                if actype not in avaliables:
                    actype = 'A320'
                newactypes.append(actype)

                # populate fuel flow model
                es = self.coeff.acs_fixwing[actype]['engines']
                e = es[list(es.keys())[0]]
                fficaos.append([e['ff_idl'], e['ff_app'], e['ff_co'], e['ff_to']])

            params = self.coeff.get_initial_values(newactypes, lifttype=coeff.LIFT_FIXWING)

            self.actype[-n:] = newactypes
            self.lifttype[-n:] = coeff.LIFT_FIXWING

            fficaos = np.array(fficaos)
            self.ffidl[-n:] = fficaos[:, 0]
            self.ffapp[-n:] = fficaos[:, 1]
            self.ffco[-n:] = fficaos[:, 2]
            self.ffto[-n:] = fficaos[:, 3]

            self.Sref[-n:] = params[:, 0]
            self.mass[-n:] = 0.5 * (params[:, 1] + params[:, 2])

            self.engnum[-n:] = params[:, 3].astype(int)
            self.engtype[-n:] = params[:, 4].astype(int)
            self.engthrust[-n:] = params[:, 5]
            self.engbpr[-n:] = params[:, 6]

        elif lifttype.lower() == 'rotor':
            newactypes = []

            for actype in bs.traf.type[-n:]:
                # convert to known aircraft type - rotor
                avaliable = list(self.coeff.acs_rotor.keys())
                if actype not in avaliable:
                    actype = 'EC35'
                newactypes.append(actype)

            params = self.coeff.get_initial_values(newactypes, lifttype=coeff.LIFT_ROTOR)
            self.lifttype[-n:] = coeff.LIFT_ROTOR
            self.mass[-n:] = 0.5 * (params[:, 1] + params[:, 2])
            self.engnum[-n:] = params[:, 3].astype(int)
            self.engpower[-n:] = params[:, 4]


    def delete(self, idx):
        super(PerfNAP, self).delete(idx)


    def update(self, simt=1):
        super(PerfNAP, self).update(simt)

        # update phase, infer from spd, roc, alt
        self.phase = ph.get(self.lifttype, bs.traf.tas, bs.traf.vs, bs.traf.alt, unit='SI')

        # update limits, based on phase change
        limits = self.__construct_lim_matrix(self.actype, self.phase)
        self.vmin = limits[:, 0]
        self.vmax = limits[:, 1]
        self.vsmin = limits[:, 2]
        self.vsmax = limits[:, 3]
        self.hmax = limits[:, 4]

        # compute thrust
        #   = number of engines x engine static thrust x thrust ratio
        idx_fixwing = np.where(self.lifttype==coeff.LIFT_FIXWING)[0]
        self.thrustratio[idx_fixwing] = thrust.compute_thrust_ratio(
            self.phase[idx_fixwing], self.engbpr[idx_fixwing], bs.traf.tas[idx_fixwing], bs.traf.alt[idx_fixwing]
        )
        self.thrust[idx_fixwing] = self.engnum[idx_fixwing] * self.engthrust[idx_fixwing] * self.thrustratio[idx_fixwing]

        # TODO: implement thrust computation for rotor aircraft
        # idx_rotor = np.where(self.lifttype==coeff.LIFT_ROTOR)[0]
        # self.thrust[idx_rotor] = 0

        # compute fuel flow
        # self.fuelflow = thrust.compute_fuel_flow(
        #     self.thrustratio, self.engnum, self.ffidl, self.ffapp, self.ffco, self.ffto
        # )

        # update bank angle, due to phase change
        self.bank = np.where((self.phase==ph.TO) | (self.phase==ph.LD), 15, self.bank)
        self.bank = np.where((self.phase==ph.IC) | (self.phase==ph.CR) | (self.phase==ph.AP), 35, self.bank)

        return None

    def limits(self, indent_v, indent_vs, indent_h):
        """ apply limits on indent speed, vertical speed, and altitude """
        super(PerfNAP, self).limits(indent_v, indent_vs, indent_h)

        # if isinstance(self.vmin, np.ndarray):
        #     pass
        # else:
        #     self.update()

        indent_v = aero.vtas2cas(indent_v, indent_h)
        allow_v = np.where(indent_v < self.vmin, self.vmin, indent_v)
        allow_v = np.where(indent_v > self.vmax, self.vmax, indent_v)
        allow_v = aero.vcas2tas(indent_v, indent_h)

        allow_vs = np.where(indent_vs < self.vsmin, self.vsmin, indent_vs)
        allow_vs = np.where(indent_vs > self.vsmax, self.vsmax, indent_vs)

        allow_h = np.where(indent_h > self.hmax, self.hmax, indent_h)

        return allow_v, allow_vs, allow_h


    def __construct_lim_matrix(self, actypes, phases):
        """Compute limitations base on aircraft model and phases

        Args:
            actypes (String or 1D-array): aircraft type / model
            phases (int or 1D-array): aircraft flight phases

        Returns:
            2D-array: limitations [spd_min, spd_max, atl_min,
                alt_max, roc_min, roc_max]
        """

        nrow = len(actypes)
        ncol = 5

        # initialize the n_actype x 5 matrix
        limits = np.zeros((nrow, ncol))

        idx_fixwing = np.where(self.lifttype==coeff.LIFT_FIXWING)[0]
        unique_fixwing_mdls = np.unique(actypes[idx_fixwing])
        for mdl in unique_fixwing_mdls:
            # for each aircraft type construct the speed, roc, and alt limits based on phase
            limits[:, 0] = np.where((actypes==mdl) & (phases==ph.NA), 0, limits[:, 0])
            limits[:, 0] = np.where((actypes==mdl) & (phases==ph.TO), self.coeff.limits_fixwing[mdl]['vminto'], limits[:, 0])
            limits[:, 0] = np.where((actypes==mdl) & (phases==ph.IC), self.coeff.limits_fixwing[mdl]['vminic'], limits[:, 0])
            limits[:, 0] = np.where((actypes==mdl) & ((phases==ph.CL)|(phases==ph.CR)|(phases==ph.DE)), self.coeff.limits_fixwing[mdl]['vminer'], limits[:, 0])
            limits[:, 0] = np.where((actypes==mdl) & (phases==ph.AP), self.coeff.limits_fixwing[mdl]['vminap'], limits[:, 0])
            limits[:, 0] = np.where((actypes==mdl) & (phases==ph.LD), self.coeff.limits_fixwing[mdl]['vminld'], limits[:, 0])
            limits[:, 0] = np.where((actypes==mdl) & (phases==ph.GD), 0, limits[:, 0])

            limits[:, 1] = np.where((actypes==mdl) & (phases==ph.NA), self.coeff.limits_fixwing[mdl]['vmaxer'], limits[:, 1])
            limits[:, 1] = np.where((actypes==mdl) & (phases==ph.TO), self.coeff.limits_fixwing[mdl]['vmaxto'], limits[:, 1])
            limits[:, 1] = np.where((actypes==mdl) & (phases==ph.IC), self.coeff.limits_fixwing[mdl]['vmaxic'], limits[:, 1])
            limits[:, 1] = np.where((actypes==mdl) & ((phases==ph.CL)|(phases==ph.CR)|(phases==ph.DE)), self.coeff.limits_fixwing[mdl]['vmaxer'], limits[:, 1])
            limits[:, 1] = np.where((actypes==mdl) & (phases==ph.AP), self.coeff.limits_fixwing[mdl]['vmaxap'], limits[:, 1])
            limits[:, 1] = np.where((actypes==mdl) & (phases==ph.LD), self.coeff.limits_fixwing[mdl]['vmaxld'], limits[:, 1])
            limits[:, 1] = np.where((actypes==mdl) & (phases==ph.GD), self.coeff.limits_fixwing[mdl]['vmaxer'], limits[:, 1])

            limits[:, 2] = np.where((actypes==mdl), self.coeff.limits_fixwing[mdl]['vsmin'], limits[:, 2])
            limits[:, 3] = np.where((actypes==mdl), self.coeff.limits_fixwing[mdl]['vsmax'], limits[:, 3])

            limits[:, 4] = np.where((actypes==mdl), self.coeff.limits_fixwing[mdl]['hmax'], limits[:, 4])

        idx_rotor = np.where(self.lifttype==coeff.LIFT_ROTOR)[0]
        unique_rotor_mdls = np.unique(actypes[idx_rotor])
        for mdl in unique_rotor_mdls:
            limits[:, 0] = np.where((actypes==mdl), self.coeff.limits_rotor[mdl]['vmin'], limits[:, 0])
            limits[:, 1] = np.where((actypes==mdl), self.coeff.limits_rotor[mdl]['vmax'], limits[:, 1])
            limits[:, 2] = np.where((actypes==mdl), self.coeff.limits_rotor[mdl]['vsmin'], limits[:, 2])
            limits[:, 3] = np.where((actypes==mdl), self.coeff.limits_rotor[mdl]['vsmax'], limits[:, 3])
            limits[:, 4] = np.where((actypes==mdl), self.coeff.limits_rotor[mdl]['hmax'], limits[:, 4])

        return limits

    def engchange(self, acid, engid=None):
        pass

    def acceleration(self, simdt):
        # using a fix acceleration, to be modeled in future
        return 1.5
