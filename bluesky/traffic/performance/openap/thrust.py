import numpy as np
from bluesky.tools import aero
from bluesky.traffic.performance.openap import phase as ph


def compute_max_thr_ratio(phase, bpr, v, h, vs, thr0):
    """Computer the dynamic thrust based on engine bypass-ratio, static maximum
    thrust, aircraft true airspeed, and aircraft altitude

    Args:
        phase (int or 1D-array): phase of flight, option: phase.[NA, TO, IC, CL,
            CR, DE, FA, LD, GD]
        bpr (int or 1D-array): engine bypass ratio
        v (int or 1D-array): aircraft true airspeed
        h (int or 1D-array): aircraft altitude

    Returns:
        int or 1D-array: thust in N
    """

    n = len(phase)

    # ---- thrust ratio at takeoff ----
    ratio_takeoff = tr_takeoff(bpr, v, h)

    # ---- thrust ratio in flight ----
    ratio_inflight = inflight(v, h, vs, thr0)

    # thrust ratio array
    #   LD and GN assume ZERO thrust
    tr = np.ones(n) * ratio_inflight
    tr = np.where(phase == ph.GD, ratio_takeoff, tr)

    return tr


def tr_takeoff(bpr, v, h):
    """Compute thrust ration at take-off"""
    G0 = 0.0606 * bpr + 0.6337
    Mach = aero.vtas2mach(v, h)
    P0 = aero.p0
    P = aero.vpressure(h)
    PP = P / P0

    A = -0.4327 * PP ** 2 + 1.3855 * PP + 0.0472
    Z = 0.9106 * PP ** 3 - 1.7736 * PP ** 2 + 1.8697 * PP
    X = 0.1377 * PP ** 3 - 0.4374 * PP ** 2 + 1.3003 * PP

    ratio = (
        A
        - 0.377 * (1 + bpr) / np.sqrt((1 + 0.82 * bpr) * G0) * Z * Mach
        + (0.23 + 0.19 * np.sqrt(bpr)) * X * Mach ** 2
    )

    return ratio


def inflight(v, h, vs, thr0):
    """Compute thrust ration for inflight"""

    def dfunc(mratio):
        d = -0.4204 * mratio + 1.0824
        return d

    def nfunc(roc):
        n = 2.667e-05 * roc + 0.8633
        return n

    def mfunc(vratio, roc):
        m = -1.2043e-1 * vratio - 8.8889e-9 * roc ** 2 + 2.4444e-5 * roc + 4.7379e-1
        return m

    roc = np.abs(np.asarray(vs / aero.fpm))
    v = np.where(v < 10, 10, v)

    mach = aero.vtas2mach(v, h)
    vcas = aero.vtas2cas(v, h)

    p = aero.vpressure(h)
    p10 = aero.vpressure(10000 * aero.ft)
    p35 = aero.vpressure(35000 * aero.ft)

    # approximate thrust at top of climb (REF 2)
    F35 = (200 + 0.2 * thr0 / 4.448) * 4.448
    mach_ref = 0.8
    vcas_ref = aero.vmach2cas(mach_ref, 35000 * aero.ft)

    # segment 3: alt > 35000:
    d = dfunc(mach / mach_ref)
    b = (mach / mach_ref) ** (-0.11)
    ratio_seg3 = d * np.log(p / p35) + b

    # segment 2: 10000 < alt <= 35000:
    a = (vcas / vcas_ref) ** (-0.1)
    n = nfunc(roc)
    ratio_seg2 = a * (p / p35) ** (-0.355 * (vcas / vcas_ref) + n)

    # segment 1: alt <= 10000:
    F10 = F35 * a * (p10 / p35) ** (-0.355 * (vcas / vcas_ref) + n)
    m = mfunc(vcas / vcas_ref, roc)
    ratio_seg1 = m * (p / p35) + (F10 / F35 - m * (p10 / p35))

    ratio = np.where(
        h > 35000 * aero.ft,
        ratio_seg3,
        np.where(h > 10000 * aero.ft, ratio_seg2, ratio_seg1),
    )

    # convert to maximum static thrust ratio
    ratio_F0 = ratio * F35 / thr0

    return ratio_F0


def compute_eng_ff_coeff(ffidl, ffapp, ffco, ffto):
    """Compute fuel flow based on engine icao fuel flow model

    Args:
        thrust_ratio (1D-array): thrust ratio between 0 and 1
        n_engines (1D-array): number of engines on the aircraft
        ff_idl (1D-array): fuel flow - idle thrust
        ff_app (1D-array): fuel flow - approach
        ff_co (1D-array): fuel flow - climb out
        ff_to (1D-array): fuel flow - takeoff

    Returns:
        list of coeff: [a, b, c], fuel flow calc: ax^2 + bx + c
    """

    # standard fuel flow at test thrust ratios
    y = [0, ffidl, ffapp, ffco, ffto]
    x = [0, 0.07, 0.3, 0.85, 1.0]

    a, b, c = np.polyfit(x, y, 2)

    return a, b, c
