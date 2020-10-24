""" This module defines a set of standard geographic functions and constants for
    easy use in BlueSky. """
import numpy as np
from math import *
import pandas as pd
from scipy.interpolate import RectSphereBivariateSpline
import os

# Constants
nm  = 1852.  # m       1 nautical mile


def rwgs84(latd):
    """ Calculate the earths radius with WGS'84 geoid definition
        In:  lat [deg] (latitude)
        Out: R   [m]   (earth radius) """
    lat    = np.radians(latd)
    a      = 6378137.0       # [m] Major semi-axis WGS-84
    b      = 6356752.314245  # [m] Minor semi-axis WGS-84
    coslat = np.cos(lat)
    sinlat = np.sin(lat)

    an     = a * a * coslat
    bn     = b * b * sinlat
    ad     = a * coslat
    bd     = b * sinlat

    # Calculate radius in meters
    r = np.sqrt((an * an + bn * bn) / (ad * ad + bd * bd))

    return r
#------------------------------------------------------------


def rwgs84_matrix(latd):
    """ Calculate the earths radius with WGS'84 geoid definition
        In:  lat [deg] (Vector of latitudes)
        Out: R   [m]   (Vector of radii) """

    lat    = np.radians(latd)
    a      = 6378137.0       # [m] Major semi-axis WGS-84
    b      = 6356752.314245  # [m] Minor semi-axis WGS-84
    coslat = np.cos(lat)
    sinlat = np.sin(lat)
    an     = a * a * coslat
    bn     = b * b * sinlat
    ad     = a * coslat
    bd     = b * sinlat

    anan   = np.multiply(an, an)
    bnbn   = np.multiply(bn, bn)
    adad   = np.multiply(ad, ad)
    bdbd   = np.multiply(bd, bd)
    # Calculate radius in meters
    r      = np.sqrt(np.divide(anan + bnbn, adad + bdbd))

    return r


def qdrdist(latd1, lond1, latd2, lond2):
    """ Calculate bearing and distance, using WGS'84
        In:
            latd1,lond1 en latd2, lond2 [deg] :positions 1 & 2
        Out:
            qdr [deg] = heading from 1 to 2
            d [nm]    = distance from 1 to 2 in nm """

    # Haversine with average radius for direction

    # Check for hemisphere crossing,
    # when simple average would not work

    # res1 for same hemisphere
    res1 = rwgs84(0.5 * (latd1 + latd2))

    # res2 :different hemisphere
    a    = 6378137.0       # [m] Major semi-axis WGS-84
    r1   = rwgs84(latd1)
    r2   = rwgs84(latd2)
    res2 = 0.5 * (abs(latd1) * (r1 + a) + abs(latd2) * (r2 + a)) / \
        (np.maximum(0.000001,abs(latd1) + abs(latd2)))

    # Condition
    sw   = (latd1 * latd2 >= 0.)

    r    = sw * res1 + (1 - sw) * res2

    # Convert to radians
    lat1 = np.radians(latd1)
    lon1 = np.radians(lond1)
    lat2 = np.radians(latd2)
    lon2 = np.radians(lond2)

    
    #root = sin1 * sin1 + coslat1 * coslat2 * sin2 * sin2
    #d    =  2.0 * r * np.arctan2(np.sqrt(root) , np.sqrt(1.0 - root))
    # d =2.*r*np.arcsin(np.sqrt(sin1*sin1 + coslat1*coslat2*sin2*sin2))

    # Corrected to avoid "nan" at westward direction
    d = r*np.arccos(np.cos(lat1)*np.cos(lat2)*np.cos(lon2-lon1) + \
                 np.sin(lat1)*np.sin(lat2))

    # Bearing from Ref. http://www.movable-type.co.uk/scripts/latlong.html

    sin1 = np.sin(0.5 * (lat2 - lat1))
    sin2 = np.sin(0.5 * (lon2 - lon1))

    coslat1 = np.cos(lat1)
    coslat2 = np.cos(lat2)


    qdr = np.degrees(np.arctan2(np.sin(lon2 - lon1) * coslat2,
        coslat1 * np.sin(lat2) - np.sin(lat1) * coslat2 * np.cos(lon2 - lon1)))

    return qdr, d/nm


def qdrdist_matrix(lat1, lon1, lat2, lon2):
    """ Calculate bearing and distance vectors, using WGS'84
        In:
            latd1,lond1 en latd2, lond2 [deg] :positions 1 & 2 (vectors)
        Out:
            qdr [deg] = heading from 1 to 2 (matrix)
            d [nm]    = distance from 1 to 2 in nm (matrix) """
    prodla =  lat1.T * lat2
    condition = prodla < 0

    r = np.zeros(prodla.shape)
    r = np.where(condition, r, rwgs84_matrix(lat1.T + lat2))

    a = 6378137.0

    r = np.where(np.invert(condition), r, (np.divide(np.multiply
      (0.5, ((np.multiply(abs(lat1), (rwgs84_matrix(lat1)+a))).T +
         np.multiply(abs(lat2), (rwgs84_matrix(lat2)+a)))),
            (abs(lat1)).T+(abs(lat2)+(lat1 == 0.)*0.000001))))  # different hemisphere

    diff_lat = lat2-lat1.T
    diff_lon = lon2-lon1.T

    sin1 = (np.radians(diff_lat))
    sin2 = (np.radians(diff_lon))

    sinlat1 = np.sin(np.radians(lat1))
    sinlat2 = np.sin(np.radians(lat2))
    coslat1 = np.cos(np.radians(lat1))
    coslat2 = np.cos(np.radians(lat2))

    sin21 = np.mat(np.sin(sin2))
    cos21 = np.mat(np.cos(sin2))
    y = np.multiply(sin21, coslat2)

    x1 = np.multiply(coslat1.T, sinlat2)

    x2 = np.multiply(sinlat1.T, coslat2)
    x3 = np.multiply(x2, cos21)
    x = x1-x3

    qdr = np.degrees(np.arctan2(y, x))

    sin10 = np.mat(np.abs(np.sin(sin1/2.)))
    sin20 = np.mat(np.abs(np.sin(sin2/2.)))
    sin1sin1 = np.multiply(sin10, sin10)
    sin2sin2 = np.multiply(sin20, sin20)
    sqrt = sin1sin1 + np.multiply((coslat1.T * coslat2), sin2sin2)
    dist_c = np.multiply(2., np.arctan2(np.sqrt(sqrt), np.sqrt(1-sqrt)))
    dist = np.multiply(r/nm, dist_c)
    #    dist = np.multiply(2.*r, np.arcsin(sqrt))

    return qdr, dist


def latlondist(latd1, lond1, latd2, lond2):
    """ Calculates distance using haversine formulae and avaerage r from wgs'84
        Input:
              two lat/lon positions in degrees
        Out:
              distance in meters !!!! """

    # Haversine with average radius

    # Check for hemisphere crossing,
    # when simple average would not work

    # res1 for same hemisphere
    res1 = rwgs84(0.5*(latd1+latd2))

    # res2 :different hemisphere
    a = 6378137.0       # [m] Major semi-axis WGS-84
    r1 = rwgs84(latd1)
    r2 = rwgs84(latd2)
    res2  = 0.5*(abs(latd1)*(r1+a) + abs(latd2)*(r2+a)) / \
        (abs(latd1)+abs(latd2))

    # Condition
    sw = (latd1*latd2 >= 0.)

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
    d =  2.*r*np.arctan2(np.sqrt(root), np.sqrt(1.-root))

    #    d =2*r*np.arcsin(np.sqrt(sin1*sin1 + coslat1*coslat2*sin2*sin2))
    return d


def latlondist_matrix(lat1, lon1, lat2, lon2):
    """ Calculates distance using haversine formulae and avaerage r from wgs'84
        Input:
              two lat/lon position vectors in degrees
        Out:
              distance vector in meters !!!! """
    prodla =  lat1.T*lat2
    condition = prodla < 0

    r = np.zeros(len(prodla))
    r = np.where(condition, r, rwgs84_matrix(lat1.T+lat2))

    a = 6378137.0
    r = np.where(np.invert(condition), r, (np.divide(np.multiply(0.5,
        ((np.multiply(abs(lat1), (rwgs84_matrix(lat1)+a))).T +
            np.multiply(abs(lat2), (rwgs84_matrix(lat2)+a)))),
            (abs(lat1)).T+(abs(lat2)))))  # different hemisphere

    diff_lat = lat2-lat1.T
    diff_lon = lon2-lon1.T

    sin1 = (np.radians(diff_lat))
    sin2 = (np.radians(diff_lon))

    coslat1 = np.cos(np.radians(lat1))
    coslat2 = np.cos(np.radians(lat2))

    sin10 = np.mat(np.sin(sin1/2))
    sin20 = np.mat(np.sin(sin2/2))
    sin1sin1 =  np.multiply(sin10, sin10)
    sin2sin2 =  np.multiply(sin20, sin20)
    root =  sin1sin1+np.multiply((coslat1.T*coslat2), sin2sin2)

    #    dist = np.multiply(2.*r, np.arcsin(sqrt))
    dist_c =  np.multiply(2, np.arctan2(np.sqrt(root), np.sqrt(1.-root)))
    dist = np.multiply(r/nm, dist_c)

    return dist


def wgsg(latd):
    """ Gravity acceleration at a given latitude according to WGS'84 """
    geq = 9.7803   # m/s2 g at equator
    e2 = 6.694e-3  # eccentricity
    k  = 0.001932  # derived from flattening f, 1/f = 298.257223563

    sinlat = np.sin(np.radians(latd))
    g = geq*(1.0 + k*sinlat*sinlat) / np.sqrt(1.0 - e2*sinlat*sinlat)

    return g


def qdrpos(latd1, lond1, qdr, dist):
    """ Calculate vector with positions from vectors of reference position,
        bearing and distance.
        In:
             latd1,lond1  [deg]   ref position(s)
             qdr          [deg]   bearing (vector) from 1 to 2
             dist         [nm]    distance (vector) between 1 and 2
        Out:
             latd2,lond2 (IN DEGREES!)
        Ref for qdrpos: http://www.movable-type.co.uk/scripts/latlong.html """

    # Unit conversion
    R = rwgs84(latd1)/nm
    lat1 = np.radians(latd1)
    lon1 = np.radians(lond1)

    # Calculate new position
    lat2 = np.arcsin(np.sin(lat1)*np.cos(dist/R) +
              np.cos(lat1)*np.sin(dist/R)*np.cos(np.radians(qdr)))

    lon2 = lon1 + np.arctan2(np.sin(np.radians(qdr))*np.sin(dist/R)*np.cos(lat1),
                     np.cos(dist/R) - np.sin(lat1)*np.sin(lat2))
    return np.degrees(lat2), np.degrees(lon2)


def kwikdist(lata, lona, latb, lonb):
    """
    Quick and dirty dist [nm]
    In:
        lat/lon, lat/lon [deg]
    Out:
        dist [nm]
    """

    re      = 6371000.  # radius earth [m]
    dlat    = np.radians(latb - lata)
    dlon    = np.radians(lonb - lona)
    cavelat = np.cos(np.radians(lata + latb) * 0.5)

    dangle  = np.sqrt(dlat * dlat + dlon * dlon * cavelat * cavelat)
    dist    = re * dangle / nm

    return dist


def kwikdist_matrix(lata, lona, latb, lonb):
    """
    Quick and dirty dist [nm]
    In:
        lat/lon, lat/lon vectors [deg]
    Out:
        dist vector [nm]
    """

    re      = 6371000.  # readius earth [m]
    dlat    = np.radians(latb - lata.T)
    dlon    = np.radians(lonb - lona.T)
    cavelat = np.cos(np.radians(lata + latb.T) * 0.5)

    dangle  = np.sqrt(np.multiply(dlat, dlat) +
                      np.multiply(np.multiply(dlon, dlon),
                                  np.multiply(cavelat, cavelat)))
    dist    = re * dangle / nm

    return dist


def kwikqdrdist(lata, lona, latb, lonb):
    """Gives quick and dirty qdr[deg] and dist [nm]
       from lat/lon. (note: does not work well close to poles)"""

    re      = 6371000.  # radius earth [m]
    dlat    = np.radians(latb - lata)
    dlon    = np.radians(lonb - lona)
    cavelat = np.cos(np.radians(lata + latb) * 0.5)

    dangle  = np.sqrt(dlat * dlat + dlon * dlon * cavelat * cavelat)
    dist    = re * dangle / nm

    qdr     = np.degrees(np.arctan2(dlon * cavelat, dlat)) % 360.

    return qdr, dist


def kwikqdrdist_matrix(lata, lona, latb, lonb):
    """Gives quick and dirty qdr[deg] and dist [nm] matrices
       from lat/lon vectors. (note: does not work well close to poles)"""

    re      = 6371000.  # radius earth [m]
    dlat    = np.radians(latb - lata.T)
    dlon    = np.radians(lonb - lona.T)
    cavelat = np.cos(np.radians(lata + latb.T) * 0.5)

    dangle  = np.sqrt(np.multiply(dlat, dlat) +
                      np.multiply(np.multiply(dlon, dlon),
                                  np.multiply(cavelat, cavelat)))
    dist    = re * dangle / nm

    qdr     = np.degrees(np.arctan2(np.multiply(dlon, cavelat), dlat)) % 360.

    return qdr, dist

def kwikpos(latd1, lond1, qdr, dist):
    """ Fast, but quick and dirty, position calculation from vectors of reference position,
        bearing and distance using flat earth approximation
        In:
             latd1,lond1  [deg]   ref position(s)
             qdr          [deg]   bearing (vector) from 1 to 2
             dist         [nm]    distance (vector) between 1 and 2
        Out:
             latd2,lond2 [deg]
        Use for flat earth purposes e.g. flat display"""

    dx = dist*np.sin(np.radians(qdr))
    dy = dist*np.cos(np.radians(qdr))
    dlat = dy/60.
    dlon = dx/(np.maximum(0.01,60.*np.cos(np.radians(latd1))))
    latd2 = latd1 + dlat
    lond2 = lond1 + dlon
    return latd2,lond2

def magdec(latd, lond):
    """
    Gives matnetic declination (also called magnetic variation) at given 
    position, interpolated from an external data table. The interpolation is
    done using an object of scipy.interpolate.RectSphereBivariateSpline 
    interpo_dec, which is generated by the function init_interpo_dec() defined
    in the same module (geo.py). The interpo_dec object rvaluates the magnetic
    declination at any latitude and longitude (latd, lond).
    The function magdec() first checks if the object interpo_dec exists and is
    an object of scipy.interpolate.RectSphereBivariateSpline. If not found or
    not be the case, as happends when magdec is initially called, the function
    init_interpo_dec() will be called.
    The arguments of interpo_dec.ev() are 1-D arrays of latitudes and
    longitudes in radians, with latitude ranging from 0 to pi and longitude
    ranging from 0 to 2pi. 
    In: 
         latd, lond  [deg]  Position at which the magnetic declination is 
                            evaluated.
    Out:
         d_hdg       [deg]  Magnetic declination, the angle of difference
                            between true North and magnetic North. For instance,
                            if the declination at a certain point were 10 deg W
                            (-10 deg), then a compass at that location pointing
                            north (magnetic) would actually align 10 deg W of
                            true North. True North would be 10 deg E relative to
                            the magnetic North direction given by the compass. 
                            Declination varies with location and slowly changes
                            in time. Referenced from
            https://www.ngdc.noaa.gov/geomag/calculators/help/igrfgridHelp.html
                            In short, magnetic heading = true heading + d_hdg,
                            or,       true heading = magnetic heading - d_hdg.
    """

    # convert latitude and longitude to 1-D arrays required by interpo_dec.ev()
    latd = np.array([latd])
    lond = np.array([lond])
    # map latitude and longitude to the domains of the arguments of
    # interpo_dec.ev()
    latd = (latd+90.) * np.pi / 180.
    lond = (lond+180.) * np.pi / 180.

    # keep the object in the memory to avoid repeated operations
    global interpo_dec

    # check if interpo_dec exists
    try:
        interpo_dec
    except NameError:
        interpo_dec = init_interpo_dec()
        d_hdg = interpo_dec.ev(latd,lond)[0]

    # check if interpo_dec is ab object of 
    # scipy.interpolate.RectSphereBivariateSpline
    if isinstance(interpo_dec,RectSphereBivariateSpline) == False:
        interpo_dec = init_interpo_dec()
        d_hdg = interpo_dec.ev(latd,lond)[0]
    else:
        d_hdg = interpo_dec.ev(latd,lond)[0]
    
    return d_hdg


def init_interpo_dec():
    """
    Generates an object of scipy.interpolate.RectSphereBivariateSpline 
    interpo_dec required by magdec() that interpolates
    magnetic declination based on the data table calculated from the NOAA
    webpage
    https://www.ngdc.noaa.gov/geomag/calculators/magcalc.shtml#igrfgrid
    with the following input:
        Southern most lat:  90 S
        Northern most lat:  90 N
        Lat Step Size:      1.0
        Western most long:  180 W
        Eastern most long:  179 E
        Lon Step Size:      1.0
        Elevation:          Mean sea level 0 Feet
        Magnetic component: Declination
        Model:              WMM (2019-2024)
        Start Date:         2020 09 20
        End Date:           2020 09 20
        Step size:          1.0
        Result format:      CSV
    The grid size can be adjusted but the (1 deg by 1 deg) size should suffice 
    for practical purpose, as long as the the grids cover the entire Earth
    surface.    
    See docstring of geo.magdec() for more information.
    """
    
    declination = pd.read_csv('./declination_sealevel.csv',\
        comment='#',usecols=[1,2,4],header=None, names=['lat','lon','dec'])
    index_last = len(declination.index)
    lat_1st = declination.iloc[0,0]
    lon_1st = declination.iloc[0,1]
    num_lon = (declination.lat == lat_1st).sum()
    num_lat = (declination.lon == lon_1st).sum()
    lat_last = declination.iloc[index_last-1,0]
    lon_last = declination.iloc[index_last-1,1]
    lat_x = np.linspace(lat_last, lat_1st, num_lat) * np.pi / 180.
    lon_y = np.linspace(lon_1st, lon_last, num_lon) * np.pi / 180.
    distinct_lat = np.unique(declination.lat)
    dec_xy = np.zeros([1,num_lon])
    for val in distinct_lat:
        dec = np.array(declination.loc[declination.lat == val, 'dec'].values)\
        .reshape(1,num_lon)
        dec_xy = np.vstack((dec_xy,dec))
    
    dec_xy=dec_xy[1:,:]
    interpo_dec = RectSphereBivariateSpline(lat_x+np.pi/2., lon_y+np.pi,\
     dec_xy)
    return interpo_dec