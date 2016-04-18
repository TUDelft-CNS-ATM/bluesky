# -*- coding: utf-8 -*-
"""
Created on Tue Mar 03 16:50:19 2015

@author: Jerom Maas
"""
import numpy as np
from math import *
from aero_np import qdrdist_vector,nm,qdrpos,vtas2eas,veas2tas
from aero import ft, qdrdist

def start(dbconf):
    dbconf.CRname="MVP_LAY"

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
                
                # If the priority switch is ON (always ON for layers), it has a different meaning for Layers than for Full Mix
                # For layers -> Climbing has highest priority
                if dbconf.swprio:
#                    if dbconf.traf.id[id1] == 'AC0430' and dbconf.traf.id[id2] == 'AC0469':
#                        import pdb
#                        pdb.set_trace()
                    # If aircraft 1 is cruising, and aircraft 2 is climbing -> aircraft one solves conflict horizontally
                    if abs(dbconf.traf.vs[id1])<0.1 and dbconf.traf.alt[id2] < dbconf.traf.aalt[id2]:
                        dv[id1] = dv[id1] - dv_eby
                        dv[id1][2] = 0.0
                    # If aircraft 2 is cruising, and aircraft 1 is climbing -> aircraft two solves conflict horizontally
                    elif abs(dbconf.traf.vs[id2])<0.1 and dbconf.traf.alt[id1] < dbconf.traf.aalt[id1]:
                        dv[id2] = dv[id2] + dv_eby
                        dv[id2][2] = 0.0
                    # If aircraft 1 is cruising, and aircraft 2 is descending -> aircraft 2 solves conflict vertically
                    elif abs(dbconf.traf.vs[id1])<0.1 and  dbconf.traf.alt[id2] > dbconf.traf.aalt[id2]:
#                        dv[id1] = dv[id1] - dv_eby
                        dv[id2] = dv[id2] + dv_eby
#                        dv[id1][2] = 0.0
#                        dv[id2][0] = 0.0
#                        dv[id2][1] = 0.0
                    # If aircraft 2 is cruising, and aircraft 1 is descending -> aircraft 1 solves conflict vertically
                    elif abs(dbconf.traf.vs[id2])<0.1 and dbconf.traf.alt[id1] > dbconf.traf.aalt[id1]:
                        dv[id1] = dv[id1] - dv_eby
#                        dv[id2] = dv[id2] + dv_eby
#                        dv[id1][0] = 0.0
#                        dv[id1][1] = 0.0
#                        dv[id2][2] = 0.0
                    # C/D - C/D
                    elif dbconf.traf.alt[id1] > dbconf.traf.aalt[id1] and dbconf.traf.alt[id2] > dbconf.traf.aalt[id2]:
                        dv[id1] = dv[id1] - dv_eby
                        dv[id2] = dv[id2] + dv_eby
                    # cruising - cruising-> solved horizontally
                    else:
                        dv[id1] = dv[id1] - dv_eby
                        dv[id2] = dv[id2] + dv_eby
                        dv[id1][2] = 0.0
                        dv[id2][2] = 0.0
                else:
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
    
    # Restrict resolution direction based on swresodir
    if dbconf.swresodir == "HORIZ":
        dv[2,:] = 0.

    # the new speed vector
    newv = dv+v

    # the new speed vector in polar coordinates
    if dbconf.swresodir == "VERT":
        newtrack = (np.arctan2(v[0,:],v[1,:])*180/np.pi) %360
    else:
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
    # Condition2 ensures that aircraft do not overshoot their layer altitude
    # This is not being used for Full Mix MVP
    condition2 = dbconf.traf.avs > 0.1
    dbconf.traf.asasalt[condition] = asasalttemp[condition]
    dbconf.traf.asasalt[condition2] = dbconf.traf.aalt[condition2]

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
    # To fix descending - cruising conflicts
    if dbconf.traf.alt[id1] > dbconf.traf.aalt[id1] or dbconf.traf.alt[id2] > dbconf.traf.aalt[id2]:
        tcpa=dbconf.tinconf[id1,id2]


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
        if dbconf.swresodir == "VERT":
            dcpa[2] = 10.

    # compute the horizontal vertical components of the change in the velocity to resolve conflict
    dv1 = (iH*dcpa[0])/(abs(tcpa)*dabsH)
    dv2 = (iH*dcpa[1])/(abs(tcpa)*dabsH)
    dv3 = (iV*dcpa[2])/(abs(tcpa)*dabsV)

    # It is necessary to cap dv3 to allow implict coordination of aircraft
    # otherwise vertical conflict is solved in 1 timestep, leading to a vertical 
    # separation that is too high. If vertical dynamics are included to aircraft 
    # model in traffic.py, the below lines should be deleted
#    if dbconf.swresodir != "VERT":
#        mindv3 = -200./60.*ft
#        maxdv3 = 200/60.*ft
#        dv3 = np.maximum(mindv3,np.minimum(maxdv3,dv3))

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
	