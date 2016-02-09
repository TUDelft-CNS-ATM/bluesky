from math import *
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
#  International Standard Atmosphere
#
#   p,rho,T = atmos(h)    # atmos as function of geopotential altitude h [m]
#   a = vsound(h)         # speed of sound [m/s] as function of h[m]
#   p = pressure(h)       # calls atmos but retruns only pressure [Pa]
#   T = temperature(h)    # calculates temperature [K] (saves time rel to atmos)
#   rho = density(h)      # calls atmos but retruns only pressure [Pa]
#
#  Speed conversion at altitude h[m] in ISA:
#
# M   = tas2mach(tas,h)  # true airspeed (tas) to mach number conversion
# tas = mach2tas(M,h)    # true airspeed (tas) to mach number conversion
# tas = eas2tas(eas,h)   # equivalent airspeed to true airspeed, h in [m]
# eas = tas2eas(tas,h)   # true airspeed to equivent airspeed, h in [m]
# tas = cas2tas(cas,h)   # cas  to tas conversion both m/s, h in [m] 
# cas = tas2cas(tas,h)   # tas to cas conversion both m/s, h in [m]
# cas = mach2cas(M,h)    # Mach to cas conversion cas in m/s, h in [m]
# M   = cas2mach(cas,h)   # cas to mach copnversion cas in m/s, h in [m]

def atmos(hinput):

#
# atmos(altitude): International Standard Atmosphere calculator
#
# Input: 
#       hinput =  altitude in meters 0.0 < hinput < 84852.
# (will be clipped when outside range, integer input allowed)
#
# Output:
#       [p,rho,T]    (in SI-units: Pa, kg/m3 and K)


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
        p   = p0[i]*exp(-g0/(R*T)*(h-h0[i]))
        rho = p/(R*T)

# Calculate for temperature gradient
    else:
        T   = T0[i] + a[i]*(h-h0[i])
        p   = p0[i]*((T/T0[i])**(-g0/(a[i]*R)))
        rho = p/(R*T)

    return p,rho,T

def temp(hinput):         # hinput [m]

#
# temp (altitude): Temperature only version of ISA atmos
#
# Input: 
#       hinput =  altitude in meters 0.0 < hinput < 84852.
# (will be clipped when outside range, integer input allowed)
#
# Output:
#       T    (in SI-unit: K


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
    p,r,T = atmos(hinput)
    return p

def density(hinput):   # air density at given altitude h [m]
    p,r,T = atmos(hinput)
    return r

def vsound(hinput):  # Speed of sound for given altitude h [m]
    T = temp(hinput)
    a = sqrt(gamma*R*T)
    return a

# ---------Speed conversions---h in [m]------------------
                
def tas2mach(tas,h): # true airspeed (tas) to mach number conversion
    a = vsound(h)
    M = tas/a
    return M

def mach2tas(M,h): # true airspeed (tas) to mach number conversion
    a = vsound(h)
    tas = M*a
    return tas

def eas2tas(eas,h):   # equivalent airspeed to true airspeed
    rho = density(h)
    tas = eas*sqrt(rho0/rho)
    return tas

def tas2eas(tas,h):  # true airspeed to equivent airspeed
    rho = density(h)
    eas = tas*sqrt(rho/rho0)
    return eas

def cas2tas(cas,h):  #cas2tas conversion both m/s h in m
    p,rho,T = atmos(h)
    qdyn    = p0*((1.+rho0*cas*cas/(7.*p0))**3.5-1.)
    tas     = sqrt(7.*p/rho*((1.+qdyn/p)**(2./7.)-1.))
    return tas

def tas2cas(tas,h):  # tas2cas conversion both m/s
    p,rho,T = atmos(h)
    qdyn    = p*((1.+rho*tas*tas/(7.*p))**3.5-1.)
    cas     = sqrt(7.*p0/rho0*((qdyn/p0+1.)**(2./7.)-1.))
    return cas

def mach2cas(M,h):
    tas = mach2tas(M,h)
    cas = tas2cas(tas,h)
    return cas

def cas2mach(cas,h):
    tas = cas2tas(cas,h)
    M = tas2mach(tas,h)
    return M

#-----------------------distance calculations-----------------

def rwgs84(latd):
# From wikipedia's Earth radius:
#
# In:
#     lat [deg] = latitude
# Out:
#     r [m] = earth radius according to WGS'84 geoid
#
    lat = radians(latd)
    a = 6378137.0       # [m] Major semi-axis WGS-84
    b = 6356752.314245  # [m] Minor semi-axis WGS-84 

    coslat = cos(lat)
    sinlat = sin(lat)

    an = a*a*coslat
    bn = b*b*sinlat
    ad = a*coslat
    bd = b*sinlat
    
# Calculate radius in meters
    r = sqrt((an*an+bn*bn)/(ad*ad+bd*bd))

    return r
#------------------------------------------------------------


def latlondist(lat1,lon1,lat2,lon2):

# Input:
#       two lat/lonpositions in degrees
# Out:
#       distance in meters !!!!
#
# Calculates distance using haversine formulae and avaerage r from wgs'84

# Calculate average local earth radius
    if lat1*lat2>0.:  # same hemisphere
        r = rwgs84(0.5*(lat1+lat2))    

    else:             # different hemisphere
        a = 6378137.0       # [m] Major semi-axis WGS-84
        r1 = rwgs84(lat1)
        r2 = rwgs84(lat2)   
        r  = 0.5*(abs(lat1)*(r1+a) + abs(lat2)*(r2+a))/ \
             (abs(lat1)+abs(lat2))

# For readability first calculate sines
    sin1 = sin(0.5*radians(lat2-lat1))
    sin2 = sin(0.5*radians(lon2-lon1))

    a = sin1*sin1 + cos(radians(lat1))*cos(radians(lat2))*sin2*sin2
    c = 2. * atan2(sqrt(a), sqrt(1.-a)); 

    d = R * c

    return d


def qdrdist(latd1,lond1,latd2,lond2):
# Using WGS'84 calculate (input in degrees!)
# qdr [deg] = heading from 1 to 2
# d [nm]    = distance from 1 to 2 in nm

# Haversine with average radius
# Calculate average local earth radius
    if latd1*latd1>0.:  # same hemisphere
        R = rwgs84(0.5*(latd1+latd2))    

    else:             # different hemisphere
        a = 6378137.0       # [m] Major semi-axis WGS-84
        r1 = rwgs84(latd1)
        r2 = rwgs84(latd2)
        if latd1!=latd2:
            R  = 0.5*(abs(latd1)*(r1+a) + abs(latd2)*(r2+a))/ \
                 (abs(latd1)+abs(latd2))
        else:
            R = r1

    dLat = radians(latd2-latd1)
    dLon = radians(lond2-lond1)
    lat1 = radians(latd1)
    lat2 = radians(latd2)

    a = sin(dLat/2.) * sin(dLat/2.) + \
           sin(dLon/2.) * sin(dLon/2) * cos(lat1) * cos(lat2); 
    c = 2. * atan2(sqrt(a), sqrt(1.-a)); 
    dist = R * c / nm # nm

# Bearing
    y = sin(dLon) * cos(lat2)
    x = cos(lat1)*sin(lat2) - sin(lat1)*cos(lat2)*cos(dLon)
    qdr = degrees(atan2(y, x))


    return qdr,dist

#--------------------------------------------------------------------

def wgsg(latd):

# Gravity acceleration according to WGS'84
    geq = 9.7803 # m/s2 g at equator
    e2 = 6.694e-3 # eccentricity
    k  = 0.001932 # derived from flattening f, 1/f = 298.257223563

    sinlat = sin(radians(latd))
    g = geq*(1.0 + k*sinlat*sinlat)/sqrt(1.0 - e2*sinlat*sinlat)

    return g

#--------------------------------------------------------------------
                 
def qdrpos(latd1,lond1,qdr,dist): # input in deg and nm

# In:
#      latd1,lond1  [deg]   ref position
#      qdr          [deg]   bearing from 1 to 2
# Out:
#      latd2,lond2
#
# Ref for qdrpos: http://www.movable-type.co.uk/scripts/latlong.html

# Unit conversion

    R = rwgs84(latd1)/nm
    lat1 = radians(latd1)
    lon1 = radians(lond1)

# Calculate new position 
    lat2 = asin(sin(lat1)*cos(dist/R) + 
              cos(lat1)*sin(dist/R)*cos(radians(qdr)) );

    lon2 = lon1 + atan2(sin(radians(qdr))*sin(dist/R)*cos(lat1), 
                     cos(dist/R) - sin(lat1)*sin(lat2))
    return lat2,lon2
