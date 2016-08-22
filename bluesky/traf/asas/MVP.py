# -*- coding: utf-8 -*-
"""
Created on Tue Mar 03 16:50:19 2015

@author: Jerom Maas
"""
import numpy as np
from ...tools.aero import ft


def start(dbconf):
    pass

def resolve(dbconf, traf):
    """ Resolve all current conflicts """
    
    # Check if ASAS is ON first!    
    if not dbconf.swasas:
        return

    # Initialize an array to store the resolution velocity vector for all A/C
    dv = np.zeros((traf.ntraf,3)) 

    # If possible, solve conflicts once and copy results for symmetrical conflicts
    # If that is not possible, solve each conflict twice, once for each A/C
    if not traf.ADSBtrunc and not traf.ADSBtransnoise:
        for conflict in dbconf.conflist_now:
            
            # Determine ac indexes from callsigns
            ac1, ac2 = conflict.split(" ")
            id1, id2 = traf.id2idx(ac1), traf.id2idx(ac2)
            
            # If A/C indexes are found, then apply MVP on this conflict pair
            # Then use the MVP computed resolution to subtract and add dv_mvp 
            #     to id1 and id2, respectively
            if id1 > -1 and id2 > -1:
                dv_mvp = MVP(traf, dbconf, id1, id2)
                dv[id1] = dv[id1] - dv_mvp
                dv[id2] = dv[id2] + dv_mvp                                        
    else:
        for i in range(dbconf.nconf):
            confpair = dbconf.confpairs[i]
            ac1      = confpair[0]
            ac2      = confpair[1]
            id1      = traf.id.index(ac1)
            id2      = traf.id.index(ac2)
            if id1 >-1 and id2 > -1:
                dv_mvp   = MVP(traf, dbconf, id1, id2)
                dv[id1]  = dv[id1] - dv_mvp

    # Now we have the resolution velocity vector for all A/C, cartesian coordinates
    dv = np.transpose(dv)

    # The old speed vector, cartesian coordinates
    trkrad = np.radians(traf.trk)
    v      = np.array([np.sin(trkrad)*traf.tas,\
                       np.cos(trkrad)*traf.tas,\
                       traf.vs])

    # The new speed vector, cartesian coordinates
    newv = dv+v
    
    # Compute new speed vector in polar coordinates based on desired resolution 
    # direction: horizontal or vertical or horizontal+vertical
    if dbconf.swresohoriz: # horizontal resolutions
        if dbconf.swresospd and not dbconf.swresohdg: # SPD only
            newtrack = traf.trk
            newgs    = np.sqrt(newv[0,:]**2 + newv[1,:]**2)            
            newvs    = traf.vs           
        elif dbconf.swresohdg and not dbconf.swresospd: # HDG only
            newtrack = (np.arctan2(newv[0,:],newv[1,:])*180/np.pi) %360
            newgs    = traf.gs
            newvs    = traf.vs  
        else: # SPD + HDG
            newtrack = (np.arctan2(newv[0,:],newv[1,:])*180/np.pi) %360
            newgs    = np.sqrt(newv[0,:]**2 + newv[1,:]**2)
            newvs    = traf.vs 
    elif dbconf.swresovert: # vertical resolutions
        newtrack = traf.trk
        newgs    = traf.gs
        newvs    = newv[2,:]       
    else: # horizontal + vertical
        newtrack = (np.arctan2(newv[0,:],newv[1,:])*180/np.pi) %360
        newgs    = np.sqrt(newv[0,:]**2 + newv[1,:]**2)
        newvs    = newv[2,:]
        
    # Cap the velocity
    newgscapped = np.maximum(dbconf.vmin,np.minimum(dbconf.vmax,newgs))
    
    # Cap the vertical speed
    vscapped = np.maximum(dbconf.vsmin,np.minimum(dbconf.vsmax,newvs))
    
    # Now assign resolutions to variables in the ASAS class
    dbconf.asashdg = newtrack
    dbconf.asasspd = newgscapped
    dbconf.asasvsp = vscapped
    
    # To update asasalt, tinconf is used. tinconf is a really big value if there is 
    # no conflict. If there is a conflict, tinconf will be between 0 and the lookahead
    # time. Therefore, asasalt should only be updated for those aircraft that have a 
    # tinconf that is between 0 and the lookahead time (i.e., for the ones that are 
    # in conflict). This is what the following code does:
    altCondition = dbconf.tinconf.min(axis=1) < dbconf.dtlookahead
    asasalttemp  = dbconf.asasvsp*dbconf.tinconf.min(axis=1) + traf.alt
    dbconf.asasalt[altCondition] = asasalttemp[altCondition]
    
    # If resolutions are limited in the horizontal direction, then asasalt should
    # be equal to auto pilot alt (aalt). This is to prevent a new asasalt being computed 
    # using the auto pilot vertical speed (traf.avs) using the code in line 106 (asasalttemp) when only
    # horizontal resolutions are allowed.
    dbconf.asasalt = dbconf.asasalt*(1-dbconf.swresohoriz) + traf.apalt*dbconf.swresohoriz
           
#=================================== Modified Voltage Potential ===============
           

def MVP(traf, dbconf, id1, id2):
    """Modified Voltage Potential (MVP) resolution method"""
    
    # Get distance and qdr between id1 and id2
    dist = dbconf.dist[id1,id2]
    qdr  = dbconf.qdr[id1,id2]
    
    # Convert qdr from degrees to radians
    qdr = np.radians(qdr)
   
    # Relative position vector between id1 and id2
    drel = np.array([np.sin(qdr)*dist, \
                np.cos(qdr)*dist, \
                traf.alt[id2]-traf.alt[id1]])

    # Find track of id1 and id2 in radians
    t1 = np.radians(traf.trk[id1])
    t2 = np.radians(traf.trk[id2])
        
    # Write velocities as vectors and find relative velocity vector              
    v1 = np.array([np.sin(t1)*traf.tas[id1],np.cos(t1)*traf.tas[id1],traf.vs[id1]])
    v2 = np.array([np.sin(t2)*traf.tas[id2],np.cos(t2)*traf.tas[id2],traf.vs[id2]])
    vrel = np.array(v2-v1) 
    
    # Find tcpa (or should it be tinconf, since tinconf decided whether its a conflict?)
    tcpa = dbconf.tcpa[id1,id2] # dbconf.tinconf[id1,id2]
    
    # Find horizontal and vertical distances at the tcpa
    dcpa  = drel + vrel*tcpa
    dabsH = np.sqrt(dcpa[0]*dcpa[0]+dcpa[1]*dcpa[1])
    dabsV = dcpa[2]
    	
    # Compute horizontal and vertical intrusions
    iH = dbconf.Rm - dabsH
    iV = dbconf.dhm - dabsV
        
    # If id1 and id2 are in intrusion, assume full intrusion to force max movement
    if drel[0] < dbconf.Rm or drel[1] < dbconf.Rm:
        iH = dbconf.Rm
    if drel[2] < dbconf.dhm:
        iV = dbconf.dhm
    
    # Exception handlers for head-on conflicts
    # This is done to prevent division by zero in the next step
    if dabsH <= 10.:
        dabsH = 10.
        dcpa[0] = 10.
        dcpa[1] = 10.
    if dabsV <= 10.:
        dabsV = 10. 
        if dbconf.swresovert:
            dcpa[2] = 10.

    # Compute the resolution velocity vector in all three directions
    dv1 = (iH*dcpa[0])/(abs(tcpa)*dabsH)  # abs(tcpa) since tinconf can be positive, while tcpa can be be negative (i.e.,conflcit is behind the two aircraft). A negative tcpa would direct dv in the wrong direction.
    dv2 = (iH*dcpa[1])/(abs(tcpa)*dabsH)
    dv3 = (iV*dcpa[2])/(abs(tcpa)*dabsV)    
    
    # It is necessary to cap dv3 to prevent that a vertical conflict 
    # is solved in 1 timestep, leading to a vertical separation that is too 
    # high (high vs assumed in traf). If vertical dynamics are included to 
    # aircraft  model in traffic.py, the below three lines should be deleted.
    mindv3 = -400./60.*ft # ~ 2.016 [m/s]
    maxdv3 = 400./60.*ft
    dv3 = np.maximum(mindv3,np.minimum(maxdv3,dv3))

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