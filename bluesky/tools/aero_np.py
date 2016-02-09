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
nm  = 1852. # m       1 nautical mile
lbs = 0.453592 # kg  pound mass
g0  = 9.80665 # m/s2    Sea level gravity constant
R   = 287.05 # Used in wikipedia table: checked with 11000 m 
p0 = 101325. # Pa     Sea level pressure ISA
rho0 = 1.225 # kg/m3  Sea level density ISA
T0   = 288.15 # K   Sea level temperature ISA
gamma = 1.40 # cp/cv for air
Rearth = 6371000.  # m  Average earth radius

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


def vatmos(alt):  # alt in m
    # Temp
    T = np.maximum(288.15 - 0.0065 * alt, 216.65)

# Density
    rhotrop = 1.225*(T/288.15)**4.256848030018761 
    dhstrat = np.maximum(0.,alt-11000.)

    rho = rhotrop*np.exp(-dhstrat/6341.552161) # = *g0/(287.05*216.65))

# Pressure
    p = rho*R*T

    return p,rho,T

def vtemp(alt):         # hinput [m]
# Temp
    Tstrat = np.array(len(alt)*[216.65]) # max 22 km!
    T = np.maximum(288.15-0.0065*alt,Tstrat)

    return T

# Atmos wrappings:
def vpressure(alt):          # hinput [m]
    p,r,T = vatmos(alt)
    return p

def vdensity(alt):   # air density at given altitude h [m]
    p,r,T = vatmos(alt)
    return r

def vvsound(hinput):  # Speed of sound for given altitude h [m]
    T = vtemp(hinput)
    a = np.sqrt(gamma*R*T)
    return a

# ---------Speed conversions---h in [m]------------------
                
def vtas2mach(tas,h): # true airspeed (tas) to mach number conversion
    a = vvsound(h)
    M = tas/a
    return M

def vmach2tas(M,h): # true airspeed (tas) to mach number conversion
    a = vvsound(h)
    tas = M*a
    return tas

def veas2tas(eas,h):   # equivalent airspeed to true airspeed
    rho = vdensity(h)
    tas = eas*np.sqrt(rho0/rho)
    return tas

def vtas2eas(tas,h):  # true airspeed to equivent airspeed
    rho = vdensity(h)
    eas = tas*np.sqrt(rho/rho0)
    return eas

def vcas2tas(cas,h):  #cas2tas conversion both m/s 
    p,rho,T = vatmos(h)
    qdyn    = p0*((1.+rho0*cas*cas/(7.*p0))**3.5-1.)
    tas     = np.sqrt(7.*p/rho*((1.+qdyn/p)**(2./7.)-1.))
    return tas

def vtas2cas(tas,h):  # tas2cas conversion both m/s
    p,rho,T = vatmos(h)
    qdyn    = p*((1.+rho*tas*tas/(7.*p))**3.5-1.)
    cas     = np.sqrt(7.*p0/rho0*((qdyn/p0+1.)**(2./7.)-1.))
    return cas

def vmach2cas(M,h):
    tas = vmach2tas(M,h)
    cas = vtas2cas(tas,h)
    return cas

def vcas2mach(cas,h):
    tas = vcas2tas(cas,h)
    M = vtas2mach(tas,h)
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
    lat = np.radians(latd)
    a = 6378137.0       # [m] Major semi-axis WGS-84
    b = 6356752.314245  # [m] Minor semi-axis WGS-84 
    coslat = np.cos(lat)
    sinlat = np.sin(lat)

    an = a*a*coslat
    bn = b*b*sinlat
    ad = a*coslat
    bd = b*sinlat
    
# Calculate radius in meters
    r = np.sqrt((an*an+bn*bn)/(ad*ad+bd*bd))

    return r
#------------------------------------------------------------

def rwgs84_vector(latd):
# From wikipedia's Earth radius:
#
# In:
#     lat [deg] = latitude
# Out:
#     r [m] = earth radius according to WGS'84 geoid
#
    lat = np.radians(latd)
    a = 6378137.0       # [m] Major semi-axis WGS-84
    b = 6356752.314245  # [m] Minor semi-axis WGS-84 

    an = a*a*np.cos(lat)
    bn = b*b*np.sin(lat)
    ad = a*np.cos(lat)
    bd = b*np.sin(lat)
    
    anan =  np.multiply(an,an)
    bnbn =  np.multiply(bn,bn)
    adad =  np.multiply(ad,ad)
    bdbd =  np.multiply(bd,bd)
# Calculate radius in meters
    sqrt = np.divide(anan+bnbn,adad+bdbd)
    r = np.sqrt(sqrt)

    return r
     
    
def qdrdist_vector(lat1,lon1,lat2,lon2):
    prodla =  lat1.T*lat2
    condition = prodla<0

    r = np.zeros(len(prodla))
    r = np.where(condition,r,rwgs84_vector(lat1.T+lat2))
    
    a = 6378137.0 


    r = np.where(np.invert(condition),r,(np.divide(np.multiply\
      (0.5,((np.multiply(abs(lat1),(rwgs84_vector(lat1)+a))).T  \
         + np.multiply(abs(lat2),(rwgs84_vector(lat2)+a)))), \
            (abs(lat1)).T+(abs(lat2)+(lat1==0.)*0.000001))))  #different hemisphere
    
    
    diff_lat = lat2-lat1.T
    diff_lon = lon2-lon1.T
   
    sin1 = (np.radians(diff_lat))
    sin2 = (np.radians(diff_lon))

    sinlat1 = np.sin(np.radians(lat1))
    sinlat2 = np.sin(np.radians(lat2))
    coslat1 = np.cos(np.radians(lat1))
    coslat2 = np.cos(np.radians(lat2))

    sin10 = np.mat(np.abs(np.sin(sin1/2.)))
    sin20 = np.mat(np.abs(np.sin(sin2/2.)))
    sin1sin1 =  np.multiply(sin10,sin10)
    sin2sin2 =  np.multiply(sin20,sin20)
    sqrt =  sin1sin1+np.multiply((coslat1.T*coslat2),sin2sin2)
    

    dist_c =  np.multiply(2.,np.arctan2(np.sqrt(sqrt),np.sqrt(1-sqrt)))
    dist = np.multiply(r/nm,dist_c)
#    dist = np.multiply(2.*r, np.arcsin(sqrt))

    
    sin21 = np.mat(np.sin(sin2)) 
    cos21 = np.mat(np.cos(sin2))
    y = np.multiply(sin21,coslat2)
    
    x1 = np.multiply(coslat1.T,sinlat2)
    
    x2 = np.multiply(sinlat1.T,coslat2)
    x3 = np.multiply(x2,cos21)
    x = x1-x3
    
    qdr = np.degrees(np.arctan2(y,x))    
    
    return qdr,dist
    
    
def latlondist_vector(lat1,lon1,lat2,lon2):
    prodla =  lat1.T*lat2
    condition = prodla<0

    r = np.zeros(len(prodla))
    r = np.where(condition,r,rwgs84_vector(lat1.T+lat2))
    
    a = 6378137.0 


    r = np.where(np.invert(condition),r,(np.divide(np.multiply(0.5,((np.multiply(abs(lat1),(rwgs84_vector(lat1)+a))).T + np.multiply(abs(lat2),(rwgs84_vector(lat2)+a)))), \
            (abs(lat1)).T+(abs(lat2)))))  #different hemisphere
    
    
    diff_lat = lat2-lat1.T
    diff_lon = lon2-lon1.T
   
    sin1 = (np.radians(diff_lat))
    sin2 = (np.radians(diff_lon))

    coslat1 = np.cos(np.radians(lat1))
    coslat2 = np.cos(np.radians(lat2))

    sin10 = np.mat(np.sin(sin1/2))
    sin20 = np.mat(np.sin(sin2/2))
    sin1sin1 =  np.multiply(sin10,sin10)
    sin2sin2 =  np.multiply(sin20,sin20)
    root =  sin1sin1+np.multiply((coslat1.T*coslat2),sin2sin2)
    
#    dist = np.multiply(2.*r, np.arcsin(sqrt))

    dist_c =  np.multiply(2,np.arctan2(np.sqrt(root),np.sqrt(1.-root)))
    dist = np.multiply(r/nm,dist_c)
    
    return dist

def qdrdist(latd1,lond1,latd2,lond2):

# Using WGS'84 calculate (input in degrees!)
# In:
#     latd1,lond1 en latd2, lond2 [deg] :positions 1 & 2
# qdr [deg] = heading from 1 to 2
# d [nm]    = distance from 1 to 2 in nm
# Nautical mile: 1852. used for downward
# compatibility

# Haversine with average radius

# Check for hemisphere crossing,
# when simple average would not work

# res1 for same hemisphere

    res1 = rwgs84(0.5*(latd1+latd2))     

# res2 :different hemisphere
    a = 6378137.0       # [m] Major semi-axis WGS-84
    r1 = rwgs84(latd1)
    r2 = rwgs84(latd2)   
    res2  = 0.5*(abs(latd1)*(r1+a) + abs(latd2)*(r2+a))/ \
         (abs(latd1)+abs(latd2))

# Condition
    sw = (latd1*latd2>=0.)

    r = sw*res1+(1-sw)*res2     

# Convert to radians    
    lat1 = np.radians(latd1)
    lon1 = np.radians(lond1)
    lat2 = np.radians(latd2)
    lon2 = np.radians(lond2)
  
    sin1 = np.sin(0.5*(lat2-lat1))
    sin2 = np.sin(0.5*(lon2-lon1))

    coslat1 = np.cos(lat1)
    coslat2 = np.cos(lat2)

    root = sin1*sin1 + coslat1*coslat2*sin2*sin2    
    d =  2.*r*np.arctan2(np.sqrt(root),np.sqrt(1.-root))
     
#    d =2.*r*np.arcsin(np.sqrt(sin1*sin1 + coslat1*coslat2*sin2*sin2))

# Bearing from Ref. http://www.movable-type.co.uk/scripts/latlong.html

    qdr = np.degrees(np.arctan2( np.sin(lon2-lon1) * coslat2, \
              coslat1*np.sin(lat2)-np.sin(lat1)*coslat2*np.cos(lon2-lon1)))

    return qdr,d/nm

#-----------------------------------------------------------------

def latlondist(latd1,lond1,latd2,lond2):

# Input:
#       two lat/lonpositions in degrees
# Out:
#       distance in meters !!!!
#
# Calculates distance using haversine formulae and avaerage r from wgs'84

# Calculate average local earth radius
# Using WGS'84 calculate (input in degrees!)
# In:
#     latd1,lond1 en latd2, lond2 [deg] :positions 1 & 2
# qdr [deg] = heading from 1 to 2
# d [nm]    = distance from 1 to 2 in nm
# Nautical mile: 1852. used for downward
# compatibility

# Haversine with average radius

# Check for hemisphere crossing,
# when simple average would not work

# res1 for same hemisphere

    res1 = rwgs84(0.5*(latd1+latd2))     

# res2 :different hemisphere
    a = 6378137.0       # [m] Major semi-axis WGS-84
    r1 = rwgs84(latd1)
    r2 = rwgs84(latd2)   
    res2  = 0.5*(abs(latd1)*(r1+a) + abs(latd2)*(r2+a))/ \
         (abs(latd1)+abs(latd2))

# Condition
    sw = (latd1*latd2>=0.)

    r = sw*res1+(1-sw)*res2     

# Convert to radians    
    lat1 = np.radians(latd1)
    lon1 = np.radians(lond1)
    lat2 = np.radians(latd2)
    lon2 = np.radians(lond2)
  
    sin1 = np.sin(0.5*(lat2-lat1))
    sin2 = np.sin(0.5*(lon2-lon1))

    coslat1 = np.cos(lat1)
    coslat2 = np.cos(lat2)

    root = sin1*sin1 + coslat1*coslat2*sin2*sin2    
    d =  2.*r*np.arctan2(np.sqrt(root),np.sqrt(1.-root))
    
#    d =2*r*np.arcsin(np.sqrt(sin1*sin1 + coslat1*coslat2*sin2*sin2))

# Bearing from Ref. http://www.movable-type.co.uk/scripts/latlong.html

    return d



def qdrdistold(latd1,lond1,latd2,lond2):
# Using WGS'84 calculate (input in degrees!)
# qdr [deg] = heading from 1 to 2
# d [nm]    = distance from 1 to 2 in nm

# Haversine with average radius
    r = 0.5*(rwgs84(latd1)+rwgs84(latd2))

    lat1 = radians(latd1)
    lon1 = radians(lond1)
    lat2 = radians(latd2)
    lon2 = radians(lond2)

    sin1 = sin(0.5*(lat2-lat1))
    sin2 = sin(0.5*(lon2-lon1))

    dist =  2.*r*asin(sqrt(sin1*sin1 +  \
            cos(lat1)*cos(lat2)*sin2*sin2))  / nm   # dist is in [nm]

# Bearing from Ref. http://www.movable-type.co.uk/scripts/latlong.html

    qdr = degrees(atan2( sin(lon2-lon1) * cos(lat2), \
                         cos(lat1)*sin(lat2)-sin(lat1)*cos(lat2)*cos(lon2-lon1)))

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
#      latd2,lond2 (IN DEGREES!)
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
    return degrees(lat2),degrees(lon2)
#
#la1 = np.array([52.,34.,-10.,10.])
#lo1 = np.array([4.,-12,34.,12.])
#la2 = np.array([32.,45.,12.,-10.])
#lo2=np.array([2.,4,-.1,0.])
#print "qdrdist = ",qdrdist(la1,lo1,la2,lo2)
#print latlondist(la1,lo1,la2,lo2)
