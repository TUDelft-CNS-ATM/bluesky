# -*- coding: utf-8 -*-
"""
Created on Tue Mar 03 16:50:19 2015

@author: Jerom Maas
"""
import numpy as np
from ...tools.aero import vtas2eas


def start(dbconf):
    pass

def resolve(dbconf, traf):
    if not dbconf.swasas:
        return

    # required change in velocity
    dv = np.zeros((traf.ntraf,3)) 

    #if possible, solve conflicts once and copy results for symmetrical conflicts,
    #if that is not possible, solve each conflict twice, once for each A/C
    if not traf.ADSBtrunc and not traf.ADSBtransnoise:

        for conflict in dbconf.conflist_now:

            id1,id2 = dbconf.ConflictToIndices(conflict)

            if id1 != "Fail" and id2!= "Fail":

                dv_eby = MVP(dbconf,id1,id2)

                dv[id1] = dv[id1] - dv_eby
                dv[id2] = dv[id2] + dv_eby
                                        
    else:

        for i in range(dbconf.nconf):
            confpair = dbconf.confpairs[i]
            ac1      = confpair[0]
            ac2      = confpair[1]
            id1      = traf.id.index(ac1)
            id2      = traf.id.index(ac2)
            dv_eby   = MVP(dbconf, id1, id2)
            dv[id1]  = dv[id1] - dv_eby

    # now we have the change in speed vector for each aircraft.
    dv = np.transpose(dv)

    # the old speed vector, cartesian coordinates
    trkrad = np.radians(traf.trk)
    v = np.array([np.sin(trkrad)*traf.tas,\
        np.cos(trkrad)*traf.tas,\
        traf.vs])

    # the new speed vector
    newv = dv+v

    # the new speed vector in polar coordinates
    newtrack = (np.arctan2(newv[0,:],newv[1,:])*180/np.pi) %360
    newgs    = np.sqrt(newv[0,:]**2 + newv[1,:]**2)
    neweas   = vtas2eas(newgs,traf.alt)
     
    
    # Cap the velocity
    neweascapped=np.maximum(dbconf.vmin,np.minimum(dbconf.vmax,neweas))
    
    # now assign in the traf class
    dbconf.asashdg = newtrack
    dbconf.asasspd = neweascapped
    dbconf.asasvsp = newv[2,:]
    dbconf.asasalt = np.sign(dbconf.asasvsp) * dbconf.tinconf.min(axis=1) \
                          + traf.alt
    
#=================================== Modified Voltage Potential ===============
        
    # Resolution: MVP method 

def MVP(traf, dbconf, id1, id2):
    """Modified Voltage Potential resolution method:
      calculate change in speed"""
    dist=dbconf.dist[id1,id2]
    qdr=dbconf.qdr[id1,id2]
    
    # from degrees to radians
    qdr=np.radians(qdr)
   
   # relative position vector
    d=np.array([np.sin(qdr)*dist, \
        np.cos(qdr)*dist, \
        traf.alt[id2]-traf.alt[id1] ])

    # find track in radians
    t1=np.radians(traf.trk[id1])
    t2=np.radians(traf.trk[id2])
        
    # write velocities as vectors and find relative velocity vector              
    v1=np.array([np.sin(t1)*traf.tas[id1],np.cos(t1)*traf.tas[id1],traf.vs[id1]])
    v2=np.array([np.sin(t2)*traf.tas[id2],np.cos(t2)*traf.tas[id2],traf.vs[id2]])
    v=np.array(v2-v1) 
    
    # Find tcpa
    t=dbconf.tcpa[id1,id2]
    
    #find drel and absolute distance at tstar
    drel=d+v*t
    dabs=np.linalg.norm(drel)

    #exception: if the two aircraft are on exact collision course 
    #(passing eachother within 10 meter), change drelstar
    exactcourse = 10. #10 meter
    dif=exactcourse-dabs
    if dif>0.:
        vperp=np.array([-v[1],v[0],0.]) #rotate velocity 90 degrees in horizontal plane
        drel+=dif*vperp/np.linalg.norm(vperp) #normalize to 10 m and add to drelstar
        dabs=np.linalg.norm(drel)
        
    #intrusion at tstar
    i=dbconf.Rm-dabs
    
    # desired change in the plane's speed vector:
    dv=i*drel/(dabs*t)
    
    #Extra factor necessary! ==================================================
    # The conflict can still be solved by only horizontal
    if dbconf.Rm<dist and dabs<dist:
        erratum=np.cos(np.arcsin(dbconf.Rm/dist)-np.arcsin(dabs/dist))
        dv_plus=dv/erratum
    # Loss of horizontal separation
    else: 
        dv_plus=dv
        
    return dv_plus