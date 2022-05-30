import numpy as np
import bluesky as bs
from bluesky.tools import aero
from bluesky.tools.aero import kts, ft, fpm
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

    def __init__(self):
        super().__init__()

        self.ac_warning = False  # aircraft mdl to default warning
        self.eng_warning = False  # aircraft engine to default warning

        self.coeff = coeff.Coefficient()

        with self.settrafarrays():
            self.lifttype = np.array([])  # lift type, fixwing [1] or rotor [2]
            self.engnum = np.array([], dtype=int)  # number of engines
            self.engthrmax = np.array([])  # static engine thrust
            self.engbpr = np.array([])  # engine bypass ratio
            self.max_thrust = np.array([])  # thrust ratio at current alt spd
            self.ff_coeff_a = np.array([])  # icao fuel flows coefficient a
            self.ff_coeff_b = np.array([])  # icao fuel flows coefficient b
            self.ff_coeff_c = np.array([])  # icao fuel flows coefficient c
            self.engpower = np.array([])  # engine power, rotor ac
            self.cd0_clean = np.array([])  # Cd0, clean configuration
            self.k_clean = np.array([])  # k, clean configuration
            self.cd0_to = np.array([])  # Cd0, takeoff configuration
            self.k_to = np.array([])  # k, takeoff configuration
            self.cd0_ld = np.array([])  # Cd0, landing configuration
            self.k_ld = np.array([])  # k, landing configuration
            self.delta_cd_gear = np.array([])  # landing gear

            self.vminic = np.array([])
            self.vminer = np.array([])
            self.vminap = np.array([])
            self.vmaxic = np.array([])
            self.vmaxer = np.array([])
            self.vmaxap = np.array([])

            self.vminto = np.array([])
            self.hcross = np.array([])
            self.mmo = np.array([])

    def create(self, n=1):
        # cautious! considering multiple created aircraft with same type
        super().create(n)

        actype = bs.traf.type[-1].upper()

        # Check synonym file if not in open ap actypes
        if (actype not in self.coeff.actypes_rotor) and (
            actype not in self.coeff.dragpolar_fixwing
        ):
            if actype in self.coeff.synodict.keys():
                # warn = f"Warning: {actype} replaced by {self.coeff.synodict[actype]}"
                # print(warn)
                # bs.scr.echo(warn)
                actype = self.coeff.synodict[actype]

        # initialize aircraft / engine performance parameters
        # check fixwing or rotor, default to fixwing
        if actype in self.coeff.actypes_rotor:
            self.lifttype[-n:] = coeff.LIFT_ROTOR
            self.mass[-n:] = 0.5 * (
                self.coeff.acs_rotor[actype]["oew"]
                + self.coeff.acs_rotor[actype]["mtow"]
            )
            self.engnum[-n:] = int(self.coeff.acs_rotor[actype]["n_engines"])
            self.engpower[-n:] = self.coeff.acs_rotor[actype]["engines"][0][1]

        else:
            # convert to known aircraft type
            if actype not in self.coeff.actypes_fixwing:
                # warn = f"Warning: {actype} replaced by B744"
                # print(warn)
                # bs.scr.echo(warn)
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

        # init type specific coefficients for flight envelops
        if actype in self.coeff.limits_rotor.keys():  # rotorcraft
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

        else:
            if actype not in self.coeff.limits_fixwing.keys():
                actype = "B744"

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
            self.hcross[-n:] = self.coeff.limits_fixwing[actype]["crosscl"]
            self.mmo[-n:] = self.coeff.limits_fixwing[actype]["mmo"]

            self.cd0_clean[-n:] = self.coeff.dragpolar_fixwing[actype]["cd0_clean"]
            self.k_clean[-n:] = self.coeff.dragpolar_fixwing[actype]["k_clean"]
            self.cd0_to[-n:] = self.coeff.dragpolar_fixwing[actype]["cd0_to"]
            self.k_to[-n:] = self.coeff.dragpolar_fixwing[actype]["k_to"]
            self.cd0_ld[-n:] = self.coeff.dragpolar_fixwing[actype]["cd0_ld"]
            self.k_ld[-n:] = self.coeff.dragpolar_fixwing[actype]["k_ld"]
            self.delta_cd_gear[-n:] = self.coeff.dragpolar_fixwing[actype][
                "delta_cd_gear"
            ]

        # append update actypes, after removing unknown types
        self.actype[-n:] = [actype] * n

        # Update envelope speed limits
        mask = np.zeros_like(self.actype, dtype=bool)
        mask[-n:] = True
        self.vmin[-n:], self.vmax[-n:] = self._construct_v_limits(mask)

    def update(self, dt):
        """Periodic update function for performance calculations."""
        # update phase, infer from spd, roc, alt
        lenph1 = len(self.phase)
        self.phase = ph.get(
            self.lifttype, bs.traf.tas, bs.traf.vs, bs.traf.alt, unit="SI"
        )

        # update speed limits, based on phase change
        self.vmin, self.vmax = self._construct_v_limits()

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

        # ----- update max acceleration ----
        self.axmax = self.calc_axmax()

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
        # print(self.currentlimits())
        # print()

    def limits(self, intent_v_tas, intent_vs, intent_h, ax):
        """apply limits on indent speed, vertical speed, and altitude (called in pilot module)

        Args:
            intent_v_tas (float or 1D-array): intent true airspeed
            intent_vs (float or 1D-array): intent vertical speed
            intent_h (float or 1D-array): intent altitude
            ax (float or 1D-array): acceleration
        Returns:
            floats or 1D-arrays: Allowed TAS, Allowed vetical rate, Allowed altitude
        """
        allow_h = np.where(intent_h > self.hmax, self.hmax, intent_h)

        intent_v_cas = aero.vtas2cas(intent_v_tas, allow_h)
        allow_v_cas = np.where((intent_v_cas < self.vmin), self.vmin, intent_v_cas)
        allow_v_cas = np.where(intent_v_cas > self.vmax, self.vmax, allow_v_cas)
        allow_v_tas = aero.vcas2tas(allow_v_cas, allow_h)
        allow_v_tas = np.where(
            aero.vtas2mach(allow_v_tas, allow_h) > self.mmo,
            aero.vmach2tas(self.mmo, allow_h),
            allow_v_tas,
        )  # maximum cannot exceed MMO

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

        # corect rotercraft speed limits
        ir = np.where(self.lifttype == coeff.LIFT_ROTOR)[0]
        allow_v_tas[ir] = np.where(
            (intent_v_tas[ir] < self.vmin[ir]), self.vmin[ir], intent_v_tas[ir]
        )
        allow_v_tas[ir] = np.where(
            (intent_v_tas[ir] > self.vmax[ir]), self.vmax[ir], allow_v_tas[ir]
        )
        allow_vs[ir] = np.where(
            (intent_vs[ir] < self.vsmin[ir]), self.vsmin[ir], intent_vs[ir]
        )
        allow_vs[ir] = np.where(
            (intent_vs[ir] > self.vsmax[ir]), self.vsmax[ir], allow_vs[ir]
        )

        return allow_v_tas, allow_vs, allow_h

    def currentlimits(self, id=None):
        """Get current kinematic performance envelop.

        Args:
            id (int or 1D-array): Aircraft ID(s). Defualt to None (all aircraft).

        Returns:
            floats or 1D-arrays: Min TAS, Max TAS, Min VS, Max VS

        """
        vtasmin = aero.vcas2tas(self.vmin, bs.traf.alt)

        vtasmax = np.minimum(
            aero.vcas2tas(self.vmax, bs.traf.alt), aero.vmach2tas(self.mmo, bs.traf.alt)
        )

        if id is not None:
            return vtasmin[id], vtasmax[id], self.vsmin[id], self.vsmax[id]
        else:
            return vtasmin, vtasmax, self.vsmin, self.vsmax

    def _construct_v_limits(self, mask=True):
        """Compute speed limist base on aircraft model and flight phases

        Args:
            mask: Indices (boolean) for aircraft to construct speed limits for.
                  When no indices are passed, all aircraft are updated.

        Returns:
            2D-array: vmin, vmax
        """
        n = len(self.actype)
        vmin = np.zeros(n)
        vmax = np.zeros(n)

        ifw = np.where(np.logical_and(self.lifttype == coeff.LIFT_FIXWING, mask))[0]
        vminfw = np.zeros(len(ifw))
        vmaxfw = np.zeros(len(ifw))

        # fixwing
        # obtain flight envelope for speed, roc, and alt, based on flight phase

        # --- minimum speed ---
        vminfw = np.where(self.phase[ifw] == ph.NA, 0, vminfw)
        vminfw = np.where(self.phase[ifw] == ph.IC, self.vminic[ifw], vminfw)
        vminfw = np.where(
            (self.phase[ifw] >= ph.CL) | (self.phase[ifw] <= ph.DE), self.vminer[ifw], vminfw
        )
        vminfw = np.where(self.phase[ifw] == ph.AP, self.vminap[ifw], vminfw)
        vminfw = np.where(self.phase[ifw] == ph.GD, 0, vminfw)

        # --- maximum speed ---
        vmaxfw = np.where(self.phase[ifw] == ph.NA, self.vmaxer[ifw], vmaxfw)
        vmaxfw = np.where(self.phase[ifw] == ph.IC, self.vmaxic[ifw], vmaxfw)
        vmaxfw = np.where(
            (self.phase[ifw] >= ph.CL) | (self.phase[ifw] <= ph.DE), self.vmaxer[ifw], vmaxfw
        )
        vmaxfw = np.where(self.phase[ifw] == ph.AP, self.vmaxap[ifw], vmaxfw)
        vmaxfw = np.where(self.phase[ifw] == ph.GD, self.vmaxic[ifw], vmaxfw)

        # rotor
        ir = np.where(np.logical_and(self.lifttype == coeff.LIFT_ROTOR, mask))[0]
        vminr = self.vmin[ir]
        vmaxr = self.vmax[ir]

        vmin[ifw] = vminfw
        vmax[ifw] = vmaxfw
        vmin[ir] = vminr
        vmax[ir] = vmaxr

        if isinstance(mask, bool):
            return vmin, vmax
        return vmin[mask], vmax[mask]

    def calc_axmax(self):
        # accelerations depending on phase and wing type
        axmax_fixwing_ground = 2
        axmax_rotor = 3.5

        axmax = np.zeros(bs.traf.ntraf)

        # fix-wing, in flight
        axmax = (self.max_thrust - self.drag) / self.mass

        # fix-wing, on ground
        axmax[self.phase == ph.GD] = axmax_fixwing_ground

        # drones
        axmax[self.lifttype == coeff.LIFT_ROTOR] = axmax_rotor

        # global minumum acceleration
        axmax[axmax < 0.5] = 0.5

        return axmax

    def show_performance(self, acid):
        return (
            True,
            f"Flight phase: {ph.readable_phase(self.phase[acid])}\n"
            f"Thrust: {self.thrust[acid] / 1000:d} kN\n"
            f"Drag: {self.drag[acid] / 1000:d} kN\n"
            f"Fuel flow: {self.fuelflow[acid]:.2f} kg/s\n"
            f"Speed envelope: [{self.vmin[acid] / kts:d}, {self.vmax[acid] / kts:d}] kts\n"
            f"Vertical speed envelope: [{self.vsmin[acid] / fpm:d}, {self.vsmax[acid] / fpm:d}] fpm\n"
            f"Ceiling: {self.hmax[acid] / ft:d} ft",
        )
