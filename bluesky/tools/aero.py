""" This module defines a set of standard aerodynamic functions and constants."""
# Vectorized versions of aero conversion routines
from math import *
import numpy as np

from bluesky import settings


settings.set_variable_defaults(casmach_threshold=2.0)
# International standard atmpshere only up to 72000 ft / 22 km

#
# Constants Aeronautics
#
kts = 0.514444              # m/s  of 1 knot
ft  = 0.3048                # m    of 1 foot
fpm = ft/60.                # feet per minute
inch = 0.0254               # m    of 1 inch
sqft = 0.09290304           # 1sqft
nm  = 1852.                 # m    of 1 nautical mile
lbs = 0.453592              # kg   of 1 pound mass
g0  = 9.80665               # m/s2    Sea level gravity constant
R   = 287.05287             # Used in wikipedia table: checked with 11000 m
p0 = 101325.                # Pa     Sea level pressure ISA
rho0 = 1.225                # kg/m3  Sea level density ISA
T0   = 288.15               # K   Sea level temperature ISA
Tstrat = 216.65             # K Stratosphere temperature (until alt=22km)
gamma = 1.40                # cp/cv: adiabatic index for air
gamma1 =  0.2               # (gamma-1)/2 for air
gamma2 = 3.5                # gamma/(gamma-1) for air
beta = -0.0065              # [K/m] ISA temp gradient below tropopause
Rearth = 6371000.           # m  Average earth radius
a0  = np.sqrt(gamma*R*T0)   # sea level speed of sound ISA
casmach_thr = settings.casmach_threshold # Threshold below which speeds should
                            # be considered as Mach numbers in casormach* functions


def casmachthr(threshold:float=None):
    """ CASMACHTHR threshold

        Set a threshold below which speeds should be considered as Mach numbers
        in CRE(ATE), ADDWPT, and SPD commands. Set to zero if speeds should
        never be considered as Mach number (e.g., when simulating drones).

        Argument:
        - threshold: CAS speed threshold [m/s]
    """
    if threshold is None:
        return True, f'CASMACHTHR: The current CAS/Mach threshold is {casmach_thr} m/s ({casmach_thr / kts} kts'

    globals()['casmach_thr'] = threshold
    return True, f'CASMACHTHR: Set CAS/Mach threshold to {threshold}'


#
# Functions for aeronautics in this module
#  - physical quantities always in SI units
#  - lat,lon,course and heading in degrees
#
#  International Standard Atmosphere up to 22 km
#
#   p,rho,T = vatmos(h)    # atmos as function of geopotential altitude h [m]
#   a = vvsound(h)         # speed of sound [m/s] as function of h[m]
#   p = vpressure(h)       # calls atmos but retruns only pressure [Pa]
#   T = vtemperature(h)    # calculates temperature [K] (saves time rel to atmos)
#   rho = vdensity(h)      # calls atmos but retruns only pressure [Pa]
#
#  Speed conversion at altitude h[m] in ISA:
#
# M   = vtas2mach(tas,h)  # true airspeed (tas) to mach number conversion
# tas = vmach2tas(M,h)    # true airspeed (tas) to mach number conversion
# tas = veas2tas(eas,h)   # equivalent airspeed to true airspeed, h in [m]
# eas = vtas2eas(tas,h)   # true airspeed to equivent airspeed, h in [m]
# tas = vcas2tas(cas,h)   # cas  to tas conversion both m/s, h in [m]
# cas = vtas2cas(tas,h)   # tas to cas conversion both m/s, h in [m]
# cas = vmach2cas(M,h)    # Mach to cas conversion cas in m/s, h in [m]
# M   = vcas2mach(cas,h)   # cas to mach copnversion cas in m/s, h in [m]

# Atmosphere up to 22 km (72178 ft)


# ------------------------------------------------------------------------------
# Vectorized aero functions
# ------------------------------------------------------------------------------
def vatmos(h):
    """ Calculate atmospheric pressure, density, and temperature for a given altitude.

        Arguments:
        - h: Altitude [m]

        Returns:
        - p: Pressure [Pa]
        - rho: Density [kg / m3]
        - T: Temperature [K]
    """
    # Temp
    T = vtemp(h)

    # Density
    rhotrop = 1.225 * (T / 288.15)**4.256848030018761
    dhstrat = np.maximum(0., h - 11000.)
    rho     = rhotrop * np.exp(-dhstrat / 6341.552161)  # = *g0/(287.05*216.65))

    # Pressure
    p = rho * R * T

    return p, rho, T


def vtemp(h):
    """ Calculate atmospheric temperature for a given altitude.

        Arguments:
        - h: Altitude [m]

        Returns:
        - T: Temperature [K]
    """
    T = np.maximum(288.15 - 0.0065 * h, Tstrat)
    return T


# Atmos wrappings:
def vpressure(h):
    """ Calculate atmospheric pressure for a given altitude.

        Arguments:
        - h: Altitude [m]

        Returns:
        - p: Pressure [Pa]
    """
    p, _, _ = vatmos(h)
    return p


def vdensity(h):
    """ Calculate atmospheric density for a given altitude.

        Arguments:
        - h: Altitude [m]

        Returns:
        - rho: Density [kg / m3]
    """
    _, r, _ = vatmos(h)
    return r


def vvsound(h):
    """ Calculate the speed of sound for a given altitude.

        Arguments:
        - h: Altitude [m]

        Returns:
        - a: Speed of sound [m/s]
    """
    T = vtemp(h)
    a = np.sqrt(gamma * R * T)
    return a


# ---------Speed conversions---h in [m]------------------
def vtas2mach(tas, h):
    """ True airspeed (tas) to mach number conversion for numpy arrays.

        Arguments:
        - tas: True airspeed [m/s]
        - h: Altitude [m]

        Returns:
        - M: Mach number [-]
    """
    a = vvsound(h)
    mach = tas / a
    return mach


def vmach2tas(mach, h):
    """ Mach number to True airspeed (tas) conversion for numpy arrays.

        Arguments:
        - mach: Mach number [-]
        - h: Altitude [m]

        Returns:
        - tas: True airspeed [m/s]
    """
    a = vvsound(h)
    tas = mach * a
    return tas


def veas2tas(eas, h):
    """ Equivalent airspeed to true airspeed conversion for numpy arrays.

        Arguments:
        - eas: Equivalent airspeed [m/s]
        - h: Altitude [m]

        Returns:
        - tas: True airspeed [m/s]
    """
    rho = vdensity(h)
    tas = eas * np.sqrt(rho0 / rho)
    return tas


def vtas2eas(tas, h):
    """ True airspeed to equivalent airspeed conversion for numpy arrays.

        Arguments:
        - tas: True airspeed [m/s]
        - h: Altitude [m]

        Returns:
        - eas: Equivalent airspeed [m/s]
    """
    rho = vdensity(h)
    eas = tas * np.sqrt(rho / rho0)
    return eas


def vcas2tas(cas, h):
    """ Calibrated to true airspeed conversion for numpy arrays.

        Arguments:
        - cas: Calibrated airspeed [m/s]
        - h: Altitude [m]

        Returns:
        - tas: True airspeed [m/s]
    """
    p, rho, _ = vatmos(h)
    qdyn = p0 * ((1.0 + rho0 * cas * cas / (7.0 * p0)) ** 3.5 - 1.0)
    tas = np.sqrt(7.0 * p / rho * ((1.0 + qdyn / p) ** (2.0 / 7.0) - 1.0))

    # cope with negative speed
    tas = np.where(cas < 0, -1 * tas, tas)
    return tas


def vtas2cas(tas, h):
    """ True to calibrated airspeed conversion for numpy arrays.

        Arguments:
        - tas: True airspeed [m/s]
        - h: Altitude [m]

        Returns:
        cas: Calibrated airspeed [m/s]
    """
    p, rho, _ = vatmos(h)
    qdyn = p*((1.+rho*tas*tas/(7.*p))**3.5-1.)
    cas = np.sqrt(7.*p0/rho0*((qdyn/p0+1.)**(2./7.)-1.))

    # cope with negative speed
    cas = np.where(tas<0, -1*cas, cas)
    return cas


def vmach2cas(mach, h):
    """ Mach to calibrated airspeed conversion for numpy arrays.

        Arguments:
        - mach: Mach number [-]
        - h: Altitude [m]

        Returns:
        - cas: Calibrated airspeed [m/s]
    """
    tas = vmach2tas(mach, h)
    cas = vtas2cas(tas, h)
    return cas


def vcas2mach(cas, h):
    """ Calibrated airspeed to Mach conversion for numpy arrays.

        Arguments:
        - cas: Calibrated airspeed [m/s]
        - h: Altitude [m]

        Returns:
        - mach: Mach number [-]
    """
    tas = vcas2tas(cas, h)
    M   = vtas2mach(tas, h)
    return M

def vcasormach(spd, h):
    """ Interpret input speed as either CAS or a Mach number, and return TAS, CAS, and Mach.

        Arguments:
        - spd: Airspeed. Interpreted as Mach number [-] when its value is below the
               CAS/Mach threshold. Otherwise interpreted as CAS [m/s].
        - h: Altitude [m]

        Returns:
        - tas: True airspeed [m/s]
        - cas: Calibrated airspeed [m/s]
        - mach: Mach number [-]
    """
    ismach = np.logical_and(spd > 0.1, spd < casmach_thr)
    tas = np.where(ismach, vmach2tas(spd, h), vcas2tas(spd, h))
    cas = np.where(ismach, vtas2cas(tas, h), spd)
    mach   = np.where(ismach, spd, vtas2mach(tas, h))
    return tas, cas, mach


def vcasormach2tas(spd, h):
    """ Interpret input speed as either CAS or a Mach number, and return TAS.

        Arguments:
        - spd: Airspeed. Interpreted as Mach number [-] when its value is below the
               CAS/Mach threshold. Otherwise interpreted as CAS [m/s].
        - h: Altitude [m]

        Returns:
        - tas: True airspeed [m/s]
    """
    ismach = np.logical_and(spd > 0.1, spd < casmach_thr)
    return np.where(ismach, vmach2tas(spd, h), vcas2tas(spd, h))


def crossoveralt(cas, mach):
    """ Calculate crossover altitude for given CAS and Mach number.

        Calculates the altitude where the given CAS and Mach values
        correspond to the same true airspeed.

        (BADA User Manual 3.12, p. 12)

        Arguments:
        - cas: Calibrated airspeed [m/s]
        - mach: Mach number [-]

        Returns:
        - Altitude [m].
    """
    # Delta: pressure ratio at the transition altitude
    delta = (((1.0 + 0.5 * (gamma - 1.0) * (cas / a0) ** 2) **
                (gamma / (gamma - 1.0)) - 1.0) /
                ((1.0 + 0.5 * (gamma - 1.0) * mach ** 2) **
                (gamma / (gamma - 1.0)) - 1.0))
    # Theta: Temperature ratio at the transition altitude
    theta = delta ** (-beta * R / g0)
    return 1000.0 / 6.5 * T0 * (1.0 - theta)

# ------------------------------------------------------------------------------
# Scalar aero functions
# ------------------------------------------------------------------------------
def atmos(h):
    """ atmos(altitude): International Standard Atmosphere calculator

        Input:
              h =  altitude in meters 0.0 < h < 84852.
        (will be clipped when outside range, integer input allowed)
        Output:
              [p,rho,T]    (in SI-units: Pa, kg/m3 and K) """

    # Constants

    # Base values and gradient in table from hand-out
    # (but corrected to avoid small discontinuities at borders of layers)
    h0 = [0.0, 11000., 20000., 32000., 47000., 51000., 71000., 86852.]

    p0 = [101325.,                  # Sea level
           22631.7009099,           # 11 km
            5474.71768857,          # 20 km
             867.974468302,         # 32 km
             110.898214043,         # 47 km
              66.939,               # 51 km
               3.9564 ]             # 71 km

    T0 = [288.15,  # Sea level
          216.65,  # 11 km
          216.65,  # 20 km
          228.65,  # 32 km
          270.65,  # 47 km
          270.65,  # 51 km
          214.65]  # 71 km

    # a = lapse rate (temp gradient)
    # integer 0 indicates isothermic layer!
    a  = [-0.0065, # 0-11 km
            0,     # 11-20 km
          0.001,   # 20-32 km
          0.0028,  # 32-47 km
            0,     # 47-51 km
          -0.0028, # 51-71 km
          -0.002]  # 71-   km

    # Clip altitude to maximum!
    h = max(0.0, min(float(h), h0[-1]))

    # Find correct layer
    i = 0
    while h > h0[i+1] and i < len(h0) - 2:
        i = i+1

    # Calculate if sothermic layer
    if a[i] == 0:
        T   = T0[i]
        p   = p0[i]*exp(-g0/(R*T)*(h-h0[i]))
        rho = p/(R*T)

    # Calculate for temperature gradient
    else:
        T   = T0[i] + a[i]*(h-h0[i])
        p   = p0[i]*((T/T0[i])**(-g0/(a[i]*R)))
        rho = p/(R*T)

    return p, rho, T


def temp(h):
    """ temp (altitude): Temperature only version of ISA atmos

        Input:
              h =  altitude in meters 0.0 < h < 84852.
        (will be clipped when outside range, integer input allowed)
        Output:
              T    (in SI-unit: K """

    # Base values and gradient in table from hand-out
    # (but corrected to avoid small discontinuities at borders of layers)
    h0 = [0.0, 11000., 20000., 32000., 47000., 51000., 71000., 86852.]

    T0 = [288.15,  # Sea level
          216.65,  # 11 km
          216.65,  # 20 km
          228.65,  # 32 km
          270.65,  # 47 km
          270.65,  # 51 km
          214.65]  # 71 km

    # a = lapse rate (temp gradient)
    # integer 0 indicates isothermic layer!
    a  = [-0.0065, # 0-11 km
            0 ,    # 11-20 km
          0.001,   # 20-32 km
          0.0028,  # 32-47 km
            0 ,    # 47-51 km
          -0.0028, # 51-71 km
          -0.002]  # 71-   km

    # Clip altitude to maximum!
    h = max(0.0,min(float(h),h0[-1]))


    # Find correct layer
    i = 0
    while h>h0[i+1] and i<len(h0)-2:
        i = i+1

    # Calculate if sothermic layer
    if a[i]==0:
        T   = T0[i]

    # Calculate for temperature gradient
    else:
        T   = T0[i] + a[i]*(h-h0[i])

    return T


# Atmos wrappings:
def pressure(h):          # h [m]
    p, r, T = atmos(h)
    return p


def density(h):   # air density at given altitude h [m]
    p, r, T = atmos(h)
    return r


def vsound(h):  # Speed of sound for given altitude h [m]
    T = temp(h)
    a = sqrt(gamma*R*T)
    return a


# ---------Speed conversions---h in [m]------------------
def tas2mach(tas, h):
    """ True airspeed (tas) to mach number conversion """
    a = vsound(h)
    M = tas / a
    return M


def mach2tas(M, h):
    """ True airspeed (tas) to mach number conversion """
    a = vsound(h)
    tas = M * a
    return tas


def eas2tas(eas, h):
    """ Equivalent airspeed to true airspeed """
    rho = density(h)
    tas = eas * sqrt(rho0 / rho)
    return tas


def tas2eas(tas, h):
    """ True airspeed to equivent airspeed """
    rho = density(h)
    eas = tas * sqrt(rho / rho0)
    return eas


def cas2tas(cas, h):
    """ cas2tas conversion both m/s h in m """
    p, rho, T = atmos(h)
    qdyn = p0*((1.+rho0*cas*cas/(7.*p0))**3.5-1.)
    tas = sqrt(7.*p/rho*((1.+qdyn/p)**(2./7.)-1.))
    tas = -1 * tas if cas < 0 else tas
    return tas


def tas2cas(tas, h):
    """ tas2cas conversion both m/s """
    p, rho, T = atmos(h)
    qdyn = p*((1.+rho*tas*tas/(7.*p))**3.5-1.)
    cas = sqrt(7.*p0/rho0*((qdyn/p0+1.)**(2./7.)-1.))
    cas = -1 * cas if tas < 0 else cas
    return cas


def mach2cas(M, h):
    """ Mach to CAS conversion """
    tas = mach2tas(M, h)
    cas = tas2cas(tas, h)
    return cas


def cas2mach(cas, h):
    """ CAS Mach conversion """
    tas = cas2tas(cas, h)
    M   = tas2mach(tas, h)
    return M

def casormach(spd,h):
    if 0.1 < spd < casmach_thr:
        # Interpret spd as Mach number
        tas = mach2tas(spd, h)
        cas = mach2cas(spd, h)
        m   = spd
    else:
        # Interpret spd as CAS
        tas = cas2tas(spd,h)
        cas = spd
        m   = cas2mach(spd, h)
    return tas, cas, m

def casormach2tas(spd,h):
    if 0.1 < spd < casmach_thr:
        # Interpret spd as Mach number
        tas = mach2tas(spd, h)
    else:
        # Interpret spd as CAS
        tas = cas2tas(spd,h)
    return tas


def metres_to_feet_rounded(metres):
    """
    Converts metres to feet.
    Returns feet as rounded integer.
    """
    return int(round(metres / ft))


def metric_spd_to_knots_rounded(speed):
    """
    Converts speed in m/s to knots.
    Returns knots as rounded integer.
    """
    return int(round(speed / kts))
