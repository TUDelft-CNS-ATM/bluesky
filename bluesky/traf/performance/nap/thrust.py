import numpy as np
from bluesky.tools import aero

def compute_thrust_ratio(phase, bpr, spd, alt, unit='SI'):
    """Computer the dynamic thrust based on engine bypass-ratio, static maximum
    thrust, aircraft true airspeed, and aircraft altitude

    Args:
        phase (int or 1D-array): phase of flight, option: phase.[NA, TO, IC, CL,
            CR, DE, FA, LD, GD]
        bpr (int or 1D-array): engine bypass ratio
        tas (int or 1D-array): aircraft true airspeed (kt)
        alt (int or 1D-array): aircraft altitude (ft)

    Returns:
        int or 1D-array: thust in N
    """

    n = len(phase)

    if unit == 'EP':
        spd = spd * aero.kts
        roc = roc * aero.pfm
        alt = alt * aero.ft

    G0 = 0.0606 * bpr + 0.6337
    Mach = aero.tas2mach(v, h)
    P0 = aero.p0
    P = aero.pressure(H)
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
    tr = np.where(phase==ph.TO, ratio_takeoff, 0)
    tr = np.where(phase==ph.IC or phase==ph.CL or phase==ph.CR,
                  ratio_inflight, 0)
    tr = np.where(phase==ph.DE or phase==ph.FA,
                  ratio_idle, 0)

    return tr


def compute_fuel_flow(thrust_ratio, n_engines, fficao):
    """Compute fuel flow based on engine icao fuel flow model

    Args:
        thrust_ratio (1D-array): thrust ratio between 0 and 1
        n_engines (1D-array): number of engines on the aircraft
        fficao (2D-array): rows are
            ff_idl : fuel flow - idle thrust
            ff_ap : fuel flow - approach
            ff_co : fuel flow - climb out
            ff_to : fuel flow - takeoff

    Returns:
        float or 1D-array: Fuel flow in kg
    """

    ff_idl = fficao[:, 0]
    ff_ap = fficao[:, 1]
    ff_co = fficao[:, 2]
    ff_to = fficao[:, 3]

    # standard fuel flow at test thrust ratios
    y = [np.zeros(ff_idl.shape), ff_idl, ff_ap, ff_co, ff_to]
    x = [0, 0.07, 0.3, 0.85, 1.0]  # test thrust ratios

    ff_model = np.poly1d(np.polyfit(x, y, 2))      # fuel flow model f(T/T0)
    ff = ff_model(thrust_ratio) * n_engines
    return ff
