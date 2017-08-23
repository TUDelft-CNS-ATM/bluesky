# -*- coding: utf-8 -*-
"""
Created on Tue Mar 03 16:50:19 2015

@author: Jerom Maas
"""
import numpy as np
from bluesky.tools.aero import vtas2eas


def start(asas):
    pass


def resolve(asas, traf):
    if not asas.swasas:
        return

    # required change in velocity
    dv = np.zeros((traf.ntraf, 3))

    #if possible, solve conflicts once and copy results for symmetrical conflicts,
    #if that is not possible, solve each conflict twice, once for each A/C
    if not traf.ADSBtrunc and not traf.ADSBtransnoise:
        for conflict in asas.conflist_now:
            id1, id2 = asas.ConflictToIndices(conflict)
            if id1 != "Fail" and id2 != "Fail":
                dv_eby = Eby_straight(traf, asas, id1, id2)
                dv[id1] -= dv_eby
                dv[id2] += dv_eby
    else:
        for i in range(asas.nconf):
            confpair = asas.confpairs[i]
            ac1      = confpair[0]
            ac2      = confpair[1]
            id1      = traf.id.index(ac1)
            id2      = traf.id.index(ac2)
            dv_eby   = Eby_straight(asas, id1, id2)
            dv[id1] -= dv_eby

    # now we have the change in speed vector for each aircraft.
    dv=np.transpose(dv)
    # the old speed vector, cartesian coordinates
    trkrad=np.radians(traf.trk)
    v=np.array([np.sin(trkrad)*traf.tas,\
        np.cos(trkrad)*traf.tas,\
        traf.vs])
    # the new speed vector
    newv=dv+v

    # the new speed vector in polar coordinates
    newtrack=(np.arctan2(newv[0,:],newv[1,:])*180/np.pi) %360
    newgs=np.sqrt(newv[0,:]**2+newv[1,:]**2)
    neweas=vtas2eas(newgs,traf.alt)

    # Cap the velocity
    neweascapped=np.maximum(asas.vmin,np.minimum(asas.vmax,neweas))

    # now assign in the traf class
    asas.hdg = newtrack
    asas.tas = neweascapped
    asas.vs  = newv[2,:]
    asas.alt = np.sign(asas.vs)*1e5

#=================================== Eby Method ===============================

    # Resolution: Eby method assuming aircraft move straight forward, solving algebraically, only horizontally
def Eby_straight(traf, asas, id1, id2):
    traf = traf
    dist = asas.dist[id1,id2]
    qdr  = asas.qdr[id1,id2]
    # from degrees to radians
    qdr  = np.radians(qdr)
    # relative position vector
    d    = np.array([np.sin(qdr)*dist, \
           np.cos(qdr)*dist, \
           traf.alt[id2]-traf.alt[id1] ])

    # find track in radians
    t1 = np.radians(traf.trk[id1])
    t2 = np.radians(traf.trk[id2])

    # write velocities as vectors and find relative velocity vector
    v1=np.array([np.sin(t1)*traf.tas[id1],np.cos(t1)*traf.tas[id1],traf.vs[id1]])
    v2=np.array([np.sin(t2)*traf.tas[id2],np.cos(t2)*traf.tas[id2],traf.vs[id2]])
    v=np.array(v2-v1)
    # bear in mind: the definition of vr (relative velocity) is opposite to
    # the velocity vector in the LOS_nominal method, this just has consequences
    # for the derivation of tstar following Eby method, not more
    """
    intrusion vector:
    i(t)=self.hsep-d(t)
    d(t)=sqrt((d[0]+v[0]*t)**2+(d[1]+v[1]*t)**2)
    find max(i(t)/t)
    -write the equation out
    -take derivative, set to zero
    -simplify, take square of everything so the sqrt disappears (creates two solutions)
    -write to the form a*t**2 + b*t + c = 0
    -Solve using the quadratic formula
    """
    # These terms are used to construct a,b,c of the quadratic formula
    R2=asas.Rm**2 # in meters
    d2=np.dot(d,d) # distance vector length squared
    v2=np.dot(v,v) # velocity vector length squared
    dv=np.dot(d,v) # dot product of distance and velocity

    # Solving the quadratic formula
    a=R2*v2 - dv**2
    b=2*dv* (R2 - d2)
    c=R2*d2 - d2**2
    discrim=b**2 - 4*a*c

    if discrim<0: # if the discriminant is negative, we're done as taking the square root will result in an error
        discrim=0
    time1=(-b+np.sqrt(discrim))/(2*a)
    time2=(-b-np.sqrt(discrim))/(2*a)

    #time when the size of the conflict is largest relative to time to solve
    tstar=min(abs(time1),abs(time2))

    #find drel and absolute distance at tstar
    drelstar=d+v*tstar
    dstarabs=np.linalg.norm(drelstar)
    #exception: if the two aircraft are on exact collision course
    #(passing eachother within 10 meter), change drelstar
    exactcourse=10 #10 meter
    dif=exactcourse-dstarabs
    if dif>0:
        vperp=np.array([-v[1],v[0],0]) #rotate velocity 90 degrees in horizontal plane
        drelstar+=dif*vperp/np.linalg.norm(vperp) #normalize to 10 m and add to drelstar
        dstarabs=np.linalg.norm(drelstar)

    #intrusion at tstar
    i=asas.Rm-dstarabs

    #desired change in the plane's speed vector:
    dv=i*drelstar/(dstarabs*tstar)
    return dv
