import numpy as np
from bluesky.tools import aero
from bluesky.traffic.performance.openap import phase as ph

def compute_thrust_ratio(phase, bpr, v, h, unit='SI'):
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

    if unit == 'EP':
        v = v * aero.kts
        h = h * aero.ft

    G0 = 0.0606 * bpr + 0.6337
    Mach = aero.vtas2mach(v, h)
    P0 = aero.p0
    P = aero.vpressure(h)
    PP = P / P0

    # thrust ratio at take off
    ratio_takeoff = 1 - 0.377 * (1+bpr) / np.sqrt((1+0.82*bpr)*G0) * Mach \
               + (0.23 + 0.19 * np.sqrt(bpr)) * Mach**2

    # thrust ratio for climb and cruise
    A = -0.4327 * PP**2 + 1.3855 * PP + 0.0472
    Z = 0.9106 * PP**3 - 1.7736 * PP**2 + 1.8697 * PP
    X = 0.1377 * PP**3 - 0.4374 * PP**2 + 1.3003 * PP

    ratio_inflight = A - 0.377 * (1+bpr) / np.sqrt((1+0.82*bpr)*G0) * Z * Mach \
          + (0.23 + 0.19 * np.sqrt(bpr)) * X * Mach**2

    # thrust ratio for descent, considering 15% of inflight model thrust
    ratio_idle = 0.15 * ratio_inflight

    # thrust ratio array
    #   LD and GN assume ZERO thrust
    tr = np.zeros(n)
    tr = np.where(phase==ph.TO, ratio_takeoff, tr)
    tr = np.where((phase==ph.IC) | (phase==ph.CL) | (phase==ph.CR), ratio_inflight, tr)
    tr = np.where(phase==ph.DE, ratio_idle, tr)

    return tr


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
