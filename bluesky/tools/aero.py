# Vectorized versions of aero conversion routines
from math import *
import numpy as np

# International standard atmpshere only up to 72000 ft / 22 km

#
# Constants Aeronautics
#
kts = 0.514444 # m/s  1 knot
ft  = 0.3048  # m     1 foot
fpm = ft/60. # feet per minute
inch = 0.0254 # m     1 inch
sqft = 0.09290304 # 1sqft
nm  = 1852. # m       1 nautical mile
lbs = 0.453592 # kg  pound mass
g0  = 9.80665 # m/s2    Sea level gravity constant
R   = 287.05287 # Used in wikipedia table: checked with 11000 m 
p0 = 101325. # Pa     Sea level pressure ISA
rho0 = 1.225 # kg/m3  Sea level density ISA
T0   = 288.15 # K   Sea level temperature ISA
gamma = 1.40 # cp/cv for air
gamma1 =  0.2 # (gamma-1)/2 for air
gamma2 = 3.5  # gamma/(gamma-1) for air
beta = -0.0065 # [K/m] ISA temp gradient below tropopause 
Rearth = 6371000.  # m  Average earth radius
a0  = sqrt(gamma*R*T0)  # sea level speed of sound ISA
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
def vatmos(alt):  # alt in m
    # Temp
    T = np.maximum(288.15 - 0.0065 * alt, 216.65)

    # Density
    rhotrop = 1.225 * (T / 288.15)**4.256848030018761
    dhstrat = np.maximum(0., alt - 11000.)
    rho     = rhotrop * np.exp(-dhstrat / 6341.552161)  # = *g0/(287.05*216.65))

    # Pressure
    p = rho * R * T

    return p, rho, T


def vtemp(alt):         # hinput [m]
    # Temp
    Tstrat = np.array(len(alt) * [216.65])  # max 22 km!
    T = np.maximum(288.15 - 0.0065 * alt, Tstrat)

    return T


# Atmos wrappings:
def vpressure(alt):          # hinput [m]
    p, r, T = vatmos(alt)
    return p


def vdensity(alt):   # air density at given altitude h [m]
    p, r, T = vatmos(alt)
    return r


def vvsound(hinput):  # Speed of sound for given altitude h [m]
    T = vtemp(hinput)
    a = np.sqrt(gamma * R * T)
    return a


# ---------Speed conversions---h in [m]------------------
def vtas2mach(tas, h):
    """ True airspeed (tas) to mach number conversion """
    a = vvsound(h)
    M = tas / a
    return M


def vmach2tas(M, h):
    """ True airspeed (tas) to mach number conversion """
    a = vvsound(h)
    tas = M * a
    return tas


def veas2tas(eas, h):
    """ Equivalent airspeed to true airspeed """
    rho = vdensity(h)
    tas = eas * np.sqrt(rho0 / rho)
    return tas


def vtas2eas(tas, h):
    """ True airspeed to equivent airspeed """
    rho = vdensity(h)
    eas = tas*np.sqrt(rho / rho0)
    return eas


def vcas2tas(cas, h):
    """ cas2tas conversion both m/s """
    p, rho, T = vatmos(h)
    qdyn      = p0*((1.+rho0*cas*cas/(7.*p0))**3.5-1.)
    tas       = np.sqrt(7.*p/rho*((1.+qdyn/p)**(2./7.)-1.))
    return tas


def vtas2cas(tas, h):
    """ tas2cas conversion both m/s """
    p, rho, T = vatmos(h)
    qdyn      = p*((1.+rho*tas*tas/(7.*p))**3.5-1.)
    cas       = np.sqrt(7.*p0/rho0*((qdyn/p0+1.)**(2./7.)-1.))
    return cas


def vmach2cas(M, h):
    """ Mach to CAS conversion """
    tas = vmach2tas(M, h)
    cas = vtas2cas(tas, h)
    return cas


def vcas2mach(cas, h):
    """ CAS to Mach conversion """
    tas = vcas2tas(cas, h)
    M   = vtas2mach(tas, h)
    return M


# ------------------------------------------------------------------------------
# Scalar aero functions
# ------------------------------------------------------------------------------
def atmos(hinput):
    """ atmos(altitude): International Standard Atmosphere calculator

        Input:
              hinput =  altitude in meters 0.0 < hinput < 84852.
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
    h = max(0.0, min(float(hinput), h0[-1]))

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


def temp(hinput):
    """ temp (altitude): Temperature only version of ISA atmos

        Input:
              hinput =  altitude in meters 0.0 < hinput < 84852.
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
    h = max(0.0,min(float(hinput),h0[-1]))


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
def pressure(hinput):          # hinput [m]
    p, r, T = atmos(hinput)
    return p


def density(hinput):   # air density at given altitude h [m]
    p, r, T = atmos(hinput)
    return r


def vsound(hinput):  # Speed of sound for given altitude h [m]
    T = temp(hinput)
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
    qdyn      = p0*((1.+rho0*cas*cas/(7.*p0))**3.5-1.)
    tas       = sqrt(7.*p/rho*((1.+qdyn/p)**(2./7.)-1.))
    return tas


def tas2cas(tas, h):
    """ tas2cas conversion both m/s """
    p, rho, T = atmos(h)
    qdyn      = p*((1.+rho*tas*tas/(7.*p))**3.5-1.)
    cas       = sqrt(7.*p0/rho0*((qdyn/p0+1.)**(2./7.)-1.))
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
