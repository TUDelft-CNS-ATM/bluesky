from math import *

nm  = 1852.         # 1 nautical mile

def qdrdist(latadeg,lonadeg,latbdeg,lonbdeg):
    """Lat/lon calculations using WGS84, calculate direction from A to B [deg] and the distance in meters"""

    # conversion
    lata = radians(latadeg)
    lona = radians(lonadeg)
    latb = radians(latbdeg)
    lonb = radians(lonbdeg)

    # constants
    reqtor = 3443.92  # radius at equator in nm
    ellips = 4.4814724e-5   # ellipsoid shape of earth WGS'84

    # Calculation of unit vectors
    londif = lonb-lona
    xa=cos(lata)
    # ya=cos(dlata)*0.0)
    za = sin(lata)
    xb = cos(latb)*cos(londif)
    yb = cos(latb)*sin(londif)
    zb = sin(latb)

    zave = (za+zb)/2.

    rprime = reqtor/sqrt(1. - ellips*zave*zave)

    # distance over earth
    # prevent domain errors due to rounding errors
    sangl2 = sqrt((xb-xa)*(xb-xa)+yb*yb+(zb-za)*(zb-za))*0.5
    angle = 2.*asin(min(1.,max(-1.,sangl2)))
    dist  = angle*rprime

    #  true bearing from a to b
    cosqdr=(xa*zb - xb*za)
    sinqdr=yb

    if sinqdr*sinqdr+cosqdr*cosqdr > 0. :
        qdr=atan2(sinqdr,cosqdr)
    else:
        qdr=0.0

    if qdr <0:
        qdr=qdr+2.*pi

    # unit conversion to degrees and meters
    qdrdeg = degrees(qdr)
    distm = dist*nm
    return qdrdeg, distm


def qdrpos(latadeg,lonadeg,qdrdeg,distm):

    # constants
    reqtor = 3443.92  # radius at equator in nm
    ellips = 4.4814724e-5   # ellipsoid shape of earth WGS'84

    # conversion of units
    lata = radians(latadeg)
    lona = radians(lonadeg)
    qdr  = radians(qdrdeg)
    dist = distm/nm

    # unit vectors beacon a

    xa = cos(dble(lata))
    # ya = cos(dble(lata))*0.0
    za = sin(dble(lata))
    rprime = reqtor/sqrt(1. - ellips*za*za)
    angle  = dist/rprime
    cangle = cos(angle)
    sangle = sin(angle)

    # unit vectors beacon b
    cosqdr = cos(qdr)
    xb = xa*cangle - za*sangle*cosqdr
    yb = sangle*sin(qdr)
    zb = za*cangle + xa*sangle*cosqdr

    # lat/lon of beacon b
    londif = atan2(yb,xb)
    latb   = asin(zb)
    lonb   = lona + londif

    latbdeg = degrees(latb)
    lonbedeg = degrees(lonb)

    return latbdeg,lonbdeg


def radtopi(x):
    return fmod(x+pi,2.*pi)-pi    


def qdrqdr(latadeg,lonadeg,qdradeg,latbdeg,lonbdeg,qdrbdeg):
    """
    Compute LAT/LON of X given TRUE BEARINGs from A and B
    Distance from A is returned as function value ,but
    upon ERROR a ZERO function value will be returned.
    """

    # Unit conversion
    lata = radians(latadeg)
    lona = radians(lonadeg)
    qdra = radians(qdradeg)
    latb = radians(latbdeg)
    lonb = radians(lonbdeg)
    qdrb = radians(qdrbdeg)

    # QDRs -> azi-s
    azi1 = radtopi(qdra)
    azi2 = radtopi(qdrb)

    # Longitudes relative to A
    theta1 = 0.
    theta2 = lonb-lona

    # Local direction vectors at point 1 (A)
    ln1 = [0.]*3
    ln1[0] = -sin(lata)
    ln1[1] = 0.
    ln1[2] = cos(lata)

    le1 = [0.]*3
    le1[0] = 0.
    le1[1] = 1.
    le1[2] = 0.
 
    # Local direction vectors at point 2 (B)
    st2_ = sin(theta2)
    ct2_ = cos(theta2)
    slb_ = sin(latb)

    ln2[0] = -ct2_ * slb_
    ln2[1] = -st2_ * slb_
    ln2[2] = cos(latb)

    le2[0] = -st2_
    le2[1] = ct2_
    le2[2] = 0.

    s1 = sin(azi1)
    c1 = cos(azi1)
    s2 = sin(azi2)
    c2 = cos(azi2)

    # Orthogonal vectors, V1 and V2
    v1 = [0.]*3
    v1[0] = s1 * ln1[0] - c1 * le1[0]
    v1[1] = s1 * ln1[1] - c1 * le1[1]
    v1[2] = s1 * ln1[2] - c1 * le1[2]

    v2 = [0.]*3
    v2[0] = s2 * ln2[0] - c2 * le2[0]
    v2[1] = s2 * ln2[1] - c2 * le2[1]
    v2[2] = s2 * ln2[2] - c2 * le2[2]

    # CROSS product (in W) and normalized (in U)
    w = [0.]*3
    w[0] = v1[1] * v2[2] - v1[2] * v2[1]
    w[1] = v1[2] * v2[0] - v1[0] * v2[2]
    w[2] = v1[0] * v2[1] - v1[1] * v2[0]

    nrm = sqrt (w[0]*w[0] + w[1]*w[1] + w[2]*w[2])

    if nrm==0.:
        return 0.,0.,0.

    u = [0.]*3
    u[0] = w[0] / nrm
    u[1] = w[1] / nrm
    u[2] = w[2] / nrm

    # Find the lat/long of the two solution points and convert to degrees
    # Remember to add in the longitude of point-1 to theta!
    lat   = asin(u[2])
    theta = atan2( u[1], u[0] )
    longt = theta+lona

    # Find two solutions, current & antipodal point
    latx1  = lat
    lonx1 = radtopi(longt)

    latx2 = -lat
    lonx2 = radtopi(lonx1+pi_)

    # Check directions from A & B to see whether we have
    # valid solutions and via the other end of the world
    qdr,dist = qdrdist(degrees(lata),degrees(lona),  \
                       degrees(latx1),degrees(lonx1))
    qdra1 = radians(qdr)
    dist1 = dist/nm

    qdr,dist = qdrdist(degrees(lata),degrees(lona),  \
                       degrees(latx2),degrees(lonx2))
    qdra2 = radians(qdr)
    dist2 = dist/nm

    qdr,dist = qdrdist(degrees(latb),degrees(lonb), \
                       degrees(latx1),degrees(lonx1))
    qdrb1 = radians(qdr)

    qdr,dist = qdrdist(degrees(latb),degrees(lonb), \
                       degrees(latx2),degrees(lonx2))
    qdrb2 = radians(qdr)
 
    if (degrees(abs(radtopi(qdrb1-qdrb)))<1. and   \
       degrees(abs(radtopi(qdra1-qdra)))<1.):
        latx = latx1
        lonx = lonx1
        dqdr = dist1
 
    elif (degrees(abs(radtopi(qdrb2-qdrb)))>1. and  \
          degrees(abs(radtopi(qdra2-qdra)))>1.):
 
         latx = latx2
         lonx = lonx2
         dqdr = dist2
 
    else:
         latx = 0.
         lonx = 0.
         dqdr = 0.

    # unit conversion
    latxdeg = degrees(latx)
    lonxdeg = degrees(lonx)
    
    return latxdeg,lonxdeg,dqdr


def kwikdist(lat0,lon0,lat1,lon1):
      coslat = cos(.5*radians(lat0+lat1))
      return 60.*sqrt((lat1-lat0)*(lat1-lat0) + \
                      (lon1-lon0)*(lon1-lon0)*coslat*coslat)

