import numpy as np
import bluesky as bs
from bluesky.tools import aero
from bluesky.tools.simtime import timed_function
from bluesky.tools.trafficarrays import RegisterElementParameters
from bluesky.traffic.performance.perfbase import PerfBase
from bluesky.traffic.performance.openap import coeff, thrust
from bluesky.traffic.performance.openap import phase as ph

bs.settings.set_variable_defaults(performance_dt=1.0)


class OpenAP(PerfBase):
    """
    Open-source Aircraft Performance (OpenAP) Model

    Methods:
        create(): initialize new aircraft with performance parameters
        update(): update performance parameters
    """

    def __init__(self):
        super(OpenAP, self).__init__()

        self.ac_warning = False  # aircraft mdl to default warning
        self.eng_warning = False  # aircraft engine to default warning

        self.coeff = coeff.Coefficient()

        with RegisterElementParameters(self):
            self.actypes = np.array([], dtype=str)
            self.phase = np.array([])
            self.lifttype = np.array([])  # lift type, fixwing [1] or rotor [2]
            self.mass = np.array([])  # mass of aircraft
            self.engnum = np.array([], dtype=int)  # number of engines
            self.engthrmax = np.array([])  # static engine thrust
            self.engbpr = np.array([])  # engine bypass ratio
            self.thrust = np.array([])  # thrust ratio at current alt spd
            self.max_thrust = np.array([])  # thrust ratio at current alt spd
            self.ff_coeff_a = np.array([])  # icao fuel flows coefficient a
            self.ff_coeff_b = np.array([])  # icao fuel flows coefficient b
            self.ff_coeff_c = np.array([])  # icao fuel flows coefficient c
            self.engpower = np.array([])  # engine power, rotor ac
            self.cd0 = np.array([])  # zero drag coefficient
            self.cd0_clean = np.array([])  # Cd0, clean configuration
            self.k_clean = np.array([])  # k, clean configuration
            self.cd0_to = np.array([])  # Cd0, takeoff configuration
            self.k_to = np.array([])  # k, takeoff configuration
            self.cd0_ld = np.array([])  # Cd0, landing configuration
            self.k_ld = np.array([])  # k, landing configuration
            self.delta_cd_gear = np.array([])  # landing gear

            self.vmin = np.array([])
            self.vmax = np.array([])
            self.vminic = np.array([])
            self.vminer = np.array([])
            self.vminap = np.array([])
            self.vmaxic = np.array([])
            self.vmaxer = np.array([])
            self.vmaxap = np.array([])

            self.vsmin = np.array([])
            self.vsmax = np.array([])
            self.hmax = np.array([])
            self.axmax = np.array([])
            self.vminto = np.array([])

    def create(self, n=1):
        # cautious! considering multiple created aircraft with same type

        super(OpenAP, self).create(n)

        actype = bs.traf.type[-1].upper()

        # Check synonym file if not in open ap actypes
        if (actype not in self.coeff.actypes_rotor) and (
            actype not in self.coeff.dragpolar_fixwing
        ):
            if actype in self.coeff.synodict:
                # print(actype,"replaced by",self.coeff.synodict[actype])
                actype = self.coeff.synodict[actype]

        # check fixwing or rotor, default to fixwing
        if actype in self.coeff.actypes_rotor:
            self.lifttype[-n:] = coeff.LIFT_ROTOR
            self.mass[-n:] = 0.5 * (
                self.coeff.acs_rotor[actype]["oew"]
                + self.coeff.acs_rotor[actype]["mtow"]
            )
            self.engnum[-n:] = int(self.coeff.acs_rotor[actype]["n_engines"])
            self.engpower[-n:] = self.coeff.acs_rotor[actype]["engines"][0][
                1
            ]  # engine power (kW)

        else:
            # convert to known aircraft type
            if actype not in self.coeff.actypes_fixwing:
                actype = "B744"

            # populate fuel flow model
            es = self.coeff.acs_fixwing[actype]["engines"]
            e = es[list(es.keys())[0]]
            coeff_a, coeff_b, coeff_c = thrust.compute_eng_ff_coeff(
                e["ff_idl"], e["ff_app"], e["ff_co"], e["ff_to"]
            )

            self.lifttype[-n:] = coeff.LIFT_FIXWING

            self.Sref[-n:] = self.coeff.acs_fixwing[actype]["wa"]
            self.mass[-n:] = 0.5 * (
                self.coeff.acs_fixwing[actype]["oew"]
                + self.coeff.acs_fixwing[actype]["mtow"]
            )

            self.engnum[-n:] = int(self.coeff.acs_fixwing[actype]["n_engines"])

            self.ff_coeff_a[-n:] = coeff_a
            self.ff_coeff_b[-n:] = coeff_b
            self.ff_coeff_c[-n:] = coeff_c

            all_ac_engs = list(self.coeff.acs_fixwing[actype]["engines"].keys())
            self.engthrmax[-n:] = self.coeff.acs_fixwing[actype]["engines"][
                all_ac_engs[0]
            ]["thr"]
            self.engbpr[-n:] = self.coeff.acs_fixwing[actype]["engines"][
                all_ac_engs[0]
            ]["bpr"]

        # init type specific coefficients
        if actype in self.coeff.dragpolar_fixwing.keys():
            self.vminic[-n:] = self.coeff.limits_fixwing[actype]["vminic"]
            self.vminer[-n:] = self.coeff.limits_fixwing[actype]["vminer"]
            self.vminap[-n:] = self.coeff.limits_fixwing[actype]["vminap"]
            self.vmaxic[-n:] = self.coeff.limits_fixwing[actype]["vmaxic"]
            self.vmaxer[-n:] = self.coeff.limits_fixwing[actype]["vmaxer"]
            self.vmaxap[-n:] = self.coeff.limits_fixwing[actype]["vmaxap"]

            self.vsmin[-n:] = self.coeff.limits_fixwing[actype]["vsmin"]
            self.vsmax[-n:] = self.coeff.limits_fixwing[actype]["vsmax"]
            self.hmax[-n:] = self.coeff.limits_fixwing[actype]["hmax"]
            self.axmax[-n:] = self.coeff.limits_fixwing[actype]["axmax"]
            self.vminto[-n:] = self.coeff.limits_fixwing[actype]["vminto"]

            self.cd0_clean[-n:] = self.coeff.dragpolar_fixwing[actype]["cd0_clean"]
            self.k_clean[-n:] = self.coeff.dragpolar_fixwing[actype]["k_clean"]
            self.cd0_to[-n:] = self.coeff.dragpolar_fixwing[actype]["cd0_to"]
            self.k_to[-n:] = self.coeff.dragpolar_fixwing[actype]["k_to"]
            self.cd0_ld[-n:] = self.coeff.dragpolar_fixwing[actype]["cd0_ld"]
            self.k_ld[-n:] = self.coeff.dragpolar_fixwing[actype]["k_ld"]
            self.delta_cd_gear[-n:] = self.coeff.dragpolar_fixwing[actype][
                "delta_cd_gear"
            ]
        else:  # rotorcraft
            self.vmin[-n:] = self.coeff.limits_rotor[actype]["vmin"]
            self.vmax[-n:] = self.coeff.limits_rotor[actype]["vmax"]
            self.vsmin[-n:] = self.coeff.limits_rotor[actype]["vsmin"]
            self.vsmax[-n:] = self.coeff.limits_rotor[actype]["vsmax"]
            self.hmax[-n:] = self.coeff.limits_rotor[actype]["hmax"]

            self.vsmin[-n:] = self.coeff.limits_rotor[actype]["vsmin"]
            self.vsmax[-n:] = self.coeff.limits_rotor[actype]["vsmax"]
            self.hmax[-n:] = self.coeff.limits_rotor[actype]["hmax"]

            self.cd0_clean[-n:] = np.nan
            self.k_clean[-n:] = np.nan
            self.cd0_to[-n:] = np.nan
            self.k_to[-n:] = np.nan
            self.cd0_ld[-n:] = np.nan
            self.k_ld[-n:] = np.nan
            self.delta_cd_gear[-n:] = np.nan

        # append update actypes, after removing unkown types
        self.actypes[-n:] = [actype] * n

    @timed_function("performance", dt=bs.settings.performance_dt)
    def update(self, dt=bs.settings.performance_dt):
        super(OpenAP, self).update()

        # update phase, infer from spd, roc, alt
        lenph1 = len(self.phase)
        self.phase = ph.get(
            self.lifttype, bs.traf.tas, bs.traf.vs, bs.traf.alt, unit="SI"
        )

        # update speed limits, based on phase change
        self.vmin, self.vmax = self._construct_v_limits(self.actypes, self.phase)

        idx_fixwing = np.where(self.lifttype == coeff.LIFT_FIXWING)[0]

        # ----- compute drag -----
        # update drage coefficient based on flight phase
        self.cd0[self.phase == ph.GD] = (
            self.cd0_to[self.phase == ph.GD] + self.delta_cd_gear[self.phase == ph.GD]
        )
        self.cd0[self.phase == ph.IC] = self.cd0_to[self.phase == ph.IC]
        self.cd0[self.phase == ph.AP] = self.cd0_ld[self.phase == ph.AP]
        self.cd0[self.phase == ph.CL] = self.cd0_clean[self.phase == ph.CL]
        self.cd0[self.phase == ph.CR] = self.cd0_clean[self.phase == ph.CR]
        self.cd0[self.phase == ph.DE] = self.cd0_clean[self.phase == ph.DE]
        self.cd0[self.phase == ph.NA] = self.cd0_clean[self.phase == ph.NA]

        self.k[self.phase == ph.GD] = self.k_to[self.phase == ph.GD]
        self.k[self.phase == ph.IC] = self.k_to[self.phase == ph.IC]
        self.k[self.phase == ph.AP] = self.k_ld[self.phase == ph.AP]
        self.k[self.phase == ph.CL] = self.k_clean[self.phase == ph.CL]
        self.k[self.phase == ph.CR] = self.k_clean[self.phase == ph.CR]
        self.k[self.phase == ph.DE] = self.k_clean[self.phase == ph.DE]
        self.k[self.phase == ph.NA] = self.k_clean[self.phase == ph.NA]

        rho = aero.vdensity(bs.traf.alt[idx_fixwing])
        vtas = bs.traf.tas[idx_fixwing]
        rhovs = 0.5 * rho * vtas ** 2 * self.Sref[idx_fixwing]
        cl = self.mass[idx_fixwing] * aero.g0 / rhovs
        self.drag[idx_fixwing] = rhovs * (
            self.cd0[idx_fixwing] + self.k[idx_fixwing] * cl ** 2
        )

        # ----- compute maximum thrust -----
        max_thrustratio_fixwing = thrust.compute_max_thr_ratio(
            self.phase[idx_fixwing],
            self.engbpr[idx_fixwing],
            bs.traf.tas[idx_fixwing],
            bs.traf.alt[idx_fixwing],
            bs.traf.vs[idx_fixwing],
            self.engnum[idx_fixwing] * self.engthrmax[idx_fixwing],
        )
        self.max_thrust[idx_fixwing] = (
            max_thrustratio_fixwing
            * self.engnum[idx_fixwing]
            * self.engthrmax[idx_fixwing]
        )

        # ----- compute net thrust -----
        self.thrust[idx_fixwing] = (
            self.drag[idx_fixwing] + self.mass[idx_fixwing] * bs.traf.ax[idx_fixwing]
        )

        # ----- compute duel flow -----
        thrustratio_fixwing = self.thrust[idx_fixwing] / (
            self.engnum[idx_fixwing] * self.engthrmax[idx_fixwing]
        )
        self.fuelflow[idx_fixwing] = self.engnum[idx_fixwing] * (
            self.ff_coeff_a[idx_fixwing] * thrustratio_fixwing ** 2
            + self.ff_coeff_b[idx_fixwing] * thrustratio_fixwing
            + self.ff_coeff_c[idx_fixwing]
        )

        # TODO: implement thrust computation for rotor aircraft
        # idx_rotor = np.where(self.lifttype==coeff.LIFT_ROTOR)[0]
        # self.thrust[idx_rotor] = 0

        # update bank angle, due to phase change
        self.bank = np.where((self.phase == ph.GD), 15, self.bank)
        self.bank = np.where(
            (self.phase == ph.IC) | (self.phase == ph.CR) | (self.phase == ph.AP),
            35,
            self.bank,
        )

        # ----- debug statements -----
        # print(bs.traf.id)
        # print(self.phase)
        # print(self.thrust.astype(int))
        # print(np.round(self.fuelflow, 2))
        # print(self.drag.astype(int))
        # print()

        return None

    def limits(self, intent_v_tas, intent_vs, intent_h, ax):
        """ apply limits on indent speed, vertical speed, and altitude (called in pilot module)"""
        super(OpenAP, self).limits(intent_v_tas, intent_vs, intent_h)

        allow_h = np.where(intent_h > self.hmax, self.hmax, intent_h)

        intent_v_cas = aero.vtas2cas(intent_v_tas, allow_h)

        allow_v_cas = np.where((intent_v_cas < self.vmin), self.vmin, intent_v_cas)
        allow_v_cas = np.where(intent_v_cas > self.vmax, self.vmax, allow_v_cas)
        allow_v_tas = aero.vcas2tas(allow_v_cas, allow_h)

        vs_max_with_acc = (1 - ax / self.axmax) * self.vsmax
        allow_vs = np.where(
            (intent_vs > 0) & (intent_vs > self.vsmax), vs_max_with_acc, intent_vs
        )  # for climb with vs larger than vsmax
        allow_vs = np.where(
            (intent_vs < 0) & (intent_vs < self.vsmin), vs_max_with_acc, allow_vs
        )  # for descent with vs smaller than vsmin (negative)
        allow_vs = np.where(
            (self.phase == ph.GD) & (bs.traf.tas < self.vminto), 0, allow_vs
        )  # takeoff aircraft

        return allow_v_tas, allow_vs, allow_h

    def _construct_v_limits(self, actypes, phases):
        """Compute speed limist base on aircraft model and flight phases

        Args:
            actypes (String or 1D-array): aircraft type / model
            phases (int or 1D-array): aircraft flight phases

        Returns:
            2D-array: vmin, vmax
        """
        n = len(actypes)
        vmin = np.zeros(n)
        vmax = np.zeros(n)

        ifw = np.where(self.lifttype == coeff.LIFT_FIXWING)[0]
        vminfw = np.zeros(len(ifw))
        vmaxfw = np.zeros(len(ifw))

        # fixwing
        # obtain flight envolop for speed, roc, and alt, based on flight phase
        vminfw = np.where(phases[ifw] == ph.NA, 0, vminfw)
        vminfw = np.where(phases[ifw] == ph.IC, self.vminic[ifw], vminfw)
        vminfw = np.where(
            (phases[ifw] >= ph.CL) | (phases[ifw] <= ph.DE), self.vminer[ifw], vminfw
        )
        vminfw = np.where(phases[ifw] == ph.AP, self.vminap[ifw], vminfw)
        vminfw = np.where(phases[ifw] == ph.GD, 0, vminfw)

        vmaxfw = np.where(phases[ifw] == ph.NA, self.vmaxer[ifw], vmaxfw)
        vmaxfw = np.where(phases[ifw] == ph.IC, self.vmaxic[ifw], vmaxfw)
        vmaxfw = np.where(
            (phases[ifw] >= ph.CL) | (phases[ifw] <= ph.DE), self.vmaxer[ifw], vmaxfw
        )
        vmaxfw = np.where(phases[ifw] == ph.AP, self.vmaxap[ifw], vmaxfw)
        vmaxfw = np.where(phases[ifw] == ph.GD, self.vmaxic[ifw], vmaxfw)

        # rotor
        ir = np.where(self.lifttype == coeff.LIFT_ROTOR)[0]
        vminr = self.vmin[ir]
        vmaxr = self.vmax[ir]

        vmin[ifw] = vminfw
        vmax[ifw] = vmaxfw
        vmin[ir] = vminr
        vmax[ir] = vmaxr

        return vmin, vmax

    def engchange(self, acid, engid=None):
        bs.scr.echo("Engine change not suppoerted in OpenAP model.")
        pass

    def acceleration(self):
        # using fix accelerations depending on phase
        acc_ground = 2
        acc_air = 0.5

        accs = np.zeros(bs.traf.ntraf)
        accs[self.phase == ph.GD] = acc_ground
        accs[self.phase != ph.GD] = acc_air

        return accs

    def show_performance(self, acid):
        bs.scr.echo("Flight phase: %s" % ph.readable_phase(self.phase[acid]))
        bs.scr.echo("Thrust: %d kN" % (self.thrust[acid] / 1000))
        bs.scr.echo("Drag: %d kN" % (self.drag[acid] / 1000))
        bs.scr.echo("Fuel flow: %.2f kg/s" % self.fuelflow[acid])
        bs.scr.echo("Speed envolop: [%d, %d] m/s" % (self.vmin[acid], self.vmax[acid]))
        bs.scr.echo(
            "Vetrical speed envolop: [%d, %d] m/s"
            % (self.vsmin[acid], self.vsmax[acid])
        )
        bs.scr.echo("Ceiling: %d km" % (self.hmax[acid] / 1000))
        # self.drag.astype(int)
