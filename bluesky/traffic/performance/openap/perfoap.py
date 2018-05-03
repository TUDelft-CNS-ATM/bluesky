import numpy as np
import bluesky as bs
from bluesky.tools import aero
from bluesky.tools.trafficarrays import TrafficArrays, RegisterElementParameters
from bluesky.traffic.performance.perfbase import PerfBase
from bluesky.traffic.performance.openap import coeff, thrust
from bluesky.traffic.performance.openap import phase as ph

class OpenAP(PerfBase):
    """
    Open-source Aircraft Performance (OpenAP) Model

    Methods:
        create(): initialize new aircraft with performance parameters
        update(): update performance parameters
    """

    def __init__(self, min_update_dt=1):
        super(OpenAP, self).__init__()

        self.min_update_dt = min_update_dt    # second, minimum update dt
        self.current_sim_time = 0       # last update simulation time
        self.ac_warning = False         # aircraft mdl to default warning
        self.eng_warning = False        # aircraft engine to default warning

        self.coeff = coeff.Coefficient()
        self.n_ac = 0

        with RegisterElementParameters(self):
            self.actypes = np.array([], dtype=str)
            self.lifttype = np.array([])  # lift type, fixwing [1] or rotor [2]
            self.engnum = np.array([], dtype=int)  # number of engines
            self.engthrust = np.array([])  # static engine thrust
            self.engbpr = np.array([])  # engine bypass ratio
            self.thrustratio = np.array([])  # thrust ratio at current alt spd
            self.ff_coeff_a = np.array([])  # icao fuel flows coefficient a
            self.ff_coeff_b = np.array([])  # icao fuel flows coefficient b
            self.ff_coeff_c = np.array([])  # icao fuel flows coefficient c
            self.engpower = np.array([])    # engine power, rotor ac

    def create(self, n=1):
        # cautious! considering multiple created aircraft with same type

        super(OpenAP, self).create(n)

        actype = bs.traf.type[-n]

        # check fixwing or rotor, default fixwing if not found
        if actype in self.coeff.actypes_rotor:

            self.lifttype[-n:] = coeff.LIFT_ROTOR
            self.mass[-n:] = 0.5 * (self.coeff.acs_rotor[actype]['oew'] + self.coeff.acs_rotor[actype]['mtow'])
            self.engnum[-n:] = int(self.coeff.acs_rotor[actype]['n_engines'])
            self.engpower[-n:] = self.coeff.acs_rotor[actype]['engines'][0][1]    # engine power (kW)

        else:
            # convert to known aircraft type
            if actype not in self.coeff.actypes_fixwing:
                actype = 'A320'

            # populate fuel flow model
            es = self.coeff.acs_fixwing[actype]['engines']
            e = es[list(es.keys())[0]]
            coeff_a, coeff_b, coeff_c = thrust.compute_eng_ff_coeff(e['ff_idl'], e['ff_app'], e['ff_co'], e['ff_to'])

            self.lifttype[-n:] = coeff.LIFT_FIXWING

            self.Sref[-n:] = self.coeff.acs_fixwing[actype]['wa']
            self.mass[-n:] = 0.5 * (self.coeff.acs_fixwing[actype]['oew'] + self.coeff.acs_fixwing[actype]['mtow'])

            self.engnum[-n:] = int(self.coeff.acs_fixwing[actype]['n_engines'])

            self.ff_coeff_a[-n:] = np.ones(n) * coeff_a
            self.ff_coeff_b[-n:] = np.ones(n) * coeff_b
            self.ff_coeff_c[-n:] = np.ones(n) * coeff_c

            all_ac_engs = list(self.coeff.acs_fixwing[actype]['engines'].keys())
            self.engthrust[-n:] = self.coeff.acs_fixwing[actype]['engines'][all_ac_engs[0]]['thr']
            self.engbpr[-n:] = self.coeff.acs_fixwing[actype]['engines'][all_ac_engs[0]]['bpr']

        # append update actypes, after removing unkown types
        self.actypes[-n:] = [actype] * n
        self.n_ac += n


    def delete(self, idx):
        super(OpenAP, self).delete(idx)
        self.n_ac -= 1 


    def update(self, simt=1):
        super(OpenAP, self).update(simt)

        # update phase, infer from spd, roc, alt
        self.phase = ph.get(self.lifttype, bs.traf.tas, bs.traf.vs, bs.traf.alt, unit='SI')

        # update limits, based on phase change
        limits = self.__construct_limit_matrix(self.actypes, self.phase)
        self.vmin = limits[:, 0]
        self.vmax = limits[:, 1]
        self.vsmin = limits[:, 2]
        self.vsmax = limits[:, 3]
        self.hmax = limits[:, 4]
        self.axmax = limits[:, 5]

        # compute thrust
        #   = number of engines x engine static thrust x thrust ratio
        idx_fixwing = np.where(self.lifttype==coeff.LIFT_FIXWING)[0]
        self.thrustratio[idx_fixwing] = thrust.compute_thrust_ratio(
            self.phase[idx_fixwing], self.engbpr[idx_fixwing], bs.traf.tas[idx_fixwing], bs.traf.alt[idx_fixwing]
        )
        self.thrust[idx_fixwing] = self.engnum[idx_fixwing] * self.engthrust[idx_fixwing] * self.thrustratio[idx_fixwing]

        # compute fuel flow
        self.fuelflow = self.engnum * (self.ff_coeff_a * self.thrustratio**2 \
                                       + self.ff_coeff_b * self.thrustratio  \
                                       + self.ff_coeff_c)

        # TODO: implement thrust computation for rotor aircraft
        # idx_rotor = np.where(self.lifttype==coeff.LIFT_ROTOR)[0]
        # self.thrust[idx_rotor] = 0

        # update bank angle, due to phase change
        self.bank = np.where((self.phase==ph.TO) | (self.phase==ph.LD), 15, self.bank)
        self.bank = np.where((self.phase==ph.IC) | (self.phase==ph.CR) | (self.phase==ph.AP), 35, self.bank)

        return None

    def limits(self, intent_v_tas, intent_vs, intent_h, ax):
        """ apply limits on indent speed, vertical speed, and altitude (called in pilot module)"""
        super(OpenAP, self).limits(intent_v_tas, intent_vs, intent_h)

        # if isinstance(self.vmin, np.ndarray):
        #     pass
        # else:
        #     self.update()

        allow_h = np.where(intent_h > self.hmax, self.hmax, intent_h)

        intent_v_cas = aero.vtas2cas(intent_v_tas, allow_h)
        allow_v_cas = np.where(intent_v_cas < self.vmin, self.vmin, intent_v_cas)
        allow_v_cas = np.where(intent_v_cas > self.vmax, self.vmax, allow_v_cas)
        allow_v_tas = aero.vcas2tas(allow_v_cas, allow_h)

        vs_max_with_acc = (1 - ax / self.axmax) * self.vsmax
        allow_vs = np.where(intent_vs > self.vsmax, vs_max_with_acc, intent_vs)
        allow_vs = np.where(intent_vs < self.vsmin, self.vsmin, allow_vs)

        # print(intent_h, allow_h, self.hmax)
        # print(intent_v_cas, allow_v_cas, self.vmin, self.vmax)
        # print(self.coeff.limits_rotor['EC35'])

        return allow_v_tas, allow_vs, allow_h


    def __construct_limit_matrix(self, actypes, phases):
        """Compute limitations base on aircraft model and phases

        Args:
            actypes (String or 1D-array): aircraft type / model
            phases (int or 1D-array): aircraft flight phases

        Returns:
            2D-array: limitations [spd_min, spd_max, atl_min,
                alt_max, roc_min, roc_max]
        """

        nrow = len(actypes)
        ncol = 6

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

            limits[:, 5] = np.where((actypes==mdl), self.coeff.limits_fixwing[mdl]['axmax'], limits[:, 5])

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
        bs.scr.echo("Engine change not suppoerted in OpenAP model.")
        pass

    def acceleration(self):
        # using fix accelerations depending on phase
        acc_ground = 2
        acc_air = 0.5

        accs = np.zeros(self.n_ac)
        accs[self.phase==ph.GD] = acc_ground
        accs[self.phase!=ph.GD] = acc_air

        return accs
