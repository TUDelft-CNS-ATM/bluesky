# -*- coding: utf-8 -*-
"""
Created on Tue Mar 03 16:50:19 2015

@author: Jerom Maas
"""
import numpy as np
from aero_np import qdrdist_vector,nm,qdrpos,vtas2eas,veas2tas

def start(dbconf):
    dbconf.CRname="MVP"

def resolve(dbconf):
    if not dbconf.swasas:
        return
        
    # required change in velocity
    dv = np.zeros((dbconf.traf.ntraf,3)) 
        
    #if possible, solve conflicts once and copy results for symmetrical conflicts,
    #if that is not possible, solve each conflict twice, once for each A/C
    if not dbconf.traf.ADSBtrunc and not dbconf.traf.ADSBtransnoise:

        for conflict in dbconf.conflist_now:

            id1,id2 = dbconf.ConflictToIndices(conflict)

            if id1 != "Fail" and id2!= "Fail":

                dv_eby = MVP(dbconf,id1,id2)

                dv[id1] = dv[id1] - dv_eby
                dv[id2] = dv[id2] + dv_eby
                                        
    else:

        for i in range(dbconf.nconf):

            ac1 = dbconf.idown[i]
            ac2 = dbconf.idoth[i]

            id1 = dbconf.traf.id.index(ac1)
            id2 = dbconf.traf.id.index(ac2)

            dv_eby = MVP(dbconf,id1,id2)
            dv[id1]= dv[id1] - dv_eby
            
    # now we have the change in speed vector for each aircraft.
    dv = np.transpose(dv)

    # the old speed vector, cartesian coordinates
    trkrad = np.radians(dbconf.traf.trk)
    v = np.array([np.sin(trkrad)*dbconf.traf.tas,\
        np.cos(trkrad)*dbconf.traf.tas,\
        dbconf.traf.vs])

    # TO DO: BUILD SWITCHES FOR PRIORITIES
    # Cruising - climbing/descending
    # Cruising - Cruising (bonus: prevents vertical resolutions)
    
    # the new speed vector
    newv = dv+v

    # the new speed vector in polar coordinates
    newtrack = (np.arctan2(newv[0,:],newv[1,:])*180/np.pi) %360
    newgs    = np.sqrt(newv[0,:]**2 + newv[1,:]**2)
    neweas   = vtas2eas(newgs,dbconf.traf.alt)
    
    # Cap the velocity
    neweascapped=np.maximum(dbconf.vmin,np.minimum(dbconf.vmax,neweas))
    
    # Cap the vertical speed
    vscapped = np.maximum(dbconf.vsmin,np.minimum(dbconf.vsmax,newv[2,:]))
    
    # now assign in the traf class
    dbconf.traf.asashdg = newtrack
    dbconf.traf.asasspd = veas2tas(neweascapped,dbconf.traf.alt)
    dbconf.traf.asasvsp = vscapped
    
    # To update asasalt, tinconf is used. tinconf is a really big value if there is 
	# no conflict. If there is a conflict, tinconf will be between 0 and the lookahead
	# time. Therefore, asasalt should only be updated for those aircraft that have a 
	# tinconf that is between 0 and the lookahead time. This is what the following code does: 
    condition = dbconf.tinconf.min(axis=1)<dbconf.dtlookahead*1.2 # dtlookahead == tlook
    asasalttemp = dbconf.traf.asasvsp * dbconf.tinconf.min(axis=1) \
                          + dbconf.traf.alt
    dbconf.traf.asasalt[condition] = asasalttemp[condition]


#=================================== Modified Voltage Potential ===============
        
    # Resolution: MVP method 

def MVP(dbconf, id1, id2):
    """Modified Voltage Potential resolution method:
      calculate change in speed"""
    traf=dbconf.traf
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
    tcpa=dbconf.tcpa[id1,id2]
    
    #find horizontal and vertical distances at the tcpa
    dcpa = d+v*tcpa
    dabsH = np.sqrt(dcpa[0]*dcpa[0]+dcpa[1]*dcpa[1])
    dabsV = dcpa[2]
	
	# compute horizontal and vertical intrusions
    iH = dbconf.Rm-dabsH
    iV = dbconf.dhm-dabsV
    
	# exception handlers for head-on conflicts 
    # this is done to prevent division by zero in the next step
    if dabsH <= 10.:
        dabsH = 10.
        dcpa[0] = 10.
        dcpa[1] = 10.
    if dabsV <= 10.:
        dabsV = 10.
    
    # compute the horizontal vertical components of the change in the velocity to resolve conflict
    dv1 = (iH*dcpa[0])/(dbconf.tcpa[id1,id2]*dabsH)
    dv2 = (iH*dcpa[1])/(dbconf.tcpa[id1,id2]*dabsH)
    dv3 = (iV*dcpa[2])/(dbconf.tcpa[id1,id2]*dabsV)
      
    # combine the dv components 
    dv = np.array([dv1,dv2,dv3])

    #Extra factor necessary! ==================================================
    # Intruder outside ownship IPZ
    if dbconf.Rm<dist and dabsH<dist:
        erratum=np.cos(np.arcsin(dbconf.Rm/dist)-np.arcsin(dabsH/dist))
        dv_plus1 = dv[0]/erratum
        dv_plus2 = dv[1]/erratum
		# combine dv_plus components. Note: erratum only applies to horizontal dv components
        dv_plus = np.array([dv_plus1,dv_plus2,dv[2]])
		
    # Intruder inside ownship IPZ
    else: 
        dv_plus=dv
        
    return dv_plus
	