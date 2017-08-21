# -*- coding: utf-8 -*-
"""
Created on Tue Mar 03 16:50:19 2015

@author: Jerom Maas
"""
import numpy as np
from bluesky.tools.aero import ft, kts, fpm


def start(asas):
    pass

def resolve(asas, traf):
    """ Resolve all current conflicts """
    
    # Premble------------------------------------------------------------------
    
    # Check if ASAS is ON first!    
    if not asas.swasas:
        return
    
    # Initialize array to store the resolution velocity vector for all A/C
    dv = np.zeros((traf.ntraf,3)) 
    
    # Initialize an array to store time needed to resolve vertically 
    timesolveV = np.ones(traf.ntraf)*1e9
    
    
    # Call MVP function to resolve conflicts-----------------------------------
    
    # When ADS-B is off, calculate resolutions for both aircraft in a conflict 
    # pair in 1 go as resolution will be symetric in this case
    if not traf.adsb.truncated and not traf.adsb.transnoise:
        for conflict in asas.conflist_now:
            
            # Determine ac indexes from callsigns
            ac1      = conflict[0]
            ac2      = conflict[1]            
            id1, id2 = traf.id2idx(ac1), traf.id2idx(ac2)
            
            # If A/C indexes are found, then apply MVP on this conflict pair
            # Then use the MVP computed resolution to subtract and add dv_mvp 
            # to id1 and id2, respectively
            if id1 > -1 and id2 > -1:
                
                # Check if this conflict is in conflist_resospawncheck
                # If so, then there should be no resolution for this conflict
                # Otherwise, resolve it!
                if conflict in asas.conflist_resospawncheck:
                    dv_mvp = np.array([0.0,0.0,0.0]) # no resolution 
                else:
                    dv_mvp, tsolV = MVP(traf, asas, id1, id2)
                    # Update the time to solve vertical conflict if it is smaller than current value
                    if tsolV < timesolveV[id1]:
                        timesolveV[id1] = tsolV
                    if tsolV < timesolveV[id2]:
                        timesolveV[id2] = tsolV                         
                
                # Use priority rules if activated
                if asas.swprio:
                    dv[id1],dv[id2] = prioRules(traf, asas.priocode, dv_mvp, dv[id1], dv[id2], id1, id2)
                else: # no priority -> coopoerative resolutions
                    # since cooperative, the vertical resolution component can be halved, and then dv_mvp can be added
                    dv_mvp[2] = dv_mvp[2]/2.0
                    dv[id1]   = dv[id1] - dv_mvp
                    dv[id2]   = dv[id2] + dv_mvp 
                    # Because resolution is cooperative, then the vertical resolution can be half
                
                # Check the noreso aircraft. Nobody avoids noreso aircraft. 
                # But noreso aircraft will avoid other aircraft
                if asas.swnoreso:
                    if ac1 in asas.noresolst: # -> Then id2 does not avoid id1. 
                        dv[id2] = dv[id2] - dv_mvp
                    if ac2 in asas.noresolst: # -> Then id1 does not avoid id2. 
                        dv[id1] = dv[id1] + dv_mvp
                
                # Check the resooff aircraft. These aircraft will not do resolutions.
                if asas.swresooff:
                    if ac1 in asas.resoofflst: # -> Then id1 does not do any resolutions
                        dv[id1] = 0.0
                    if ac2 in asas.resoofflst: # -> Then id2 does not do any resolutions
                        dv[id2] = 0.0    
                    
    # If ADS-B is on, calculate resolution for each conflicting aircraft individually                                                           
    else:
        for i in range(asas.nconf):
            confpair = asas.confpairs[i]
            ac1      = confpair[0]
            ac2      = confpair[1]
            id1      = traf.id.index(ac1)
            id2      = traf.id.index(ac2)
            
            # If A/C indexes are found, then apply MVP on this conflict pair
            # Because ADSB is ON, this is done for each aircraft separately
            if id1 >-1 and id2 > -1:
                
                # Check if this conflict is in conflist_resospawncheck
                # If so, then there should be no resolution for this conflict
                # Otherwise, resolve it!
                if confpair in asas.conflist_resospawncheck:
                    dv_mvp = np.array([0.0,0.0,0.0])
                else:
                    dv_mvp, tsolV = MVP(traf, asas, id1, id2)
                    # Update the time to solve vertical conflict if it is smaller than current value
                    if tsolV < timesolveV[id1]:
                        timesolveV[id1] = tsolV
                
                # Use priority rules if activated
                if asas.swprio:
                   dv[id1], foobar = prioRules(traf, asas.priocode, dv_mvp, dv[id1], dv[id2], id1, id2) 
                else: # no priority -> cooperative resolutions
                   # since cooperative, the vertical resolution component can be halved, and then dv_mvp can be added
                   dv_mvp[2] = dv_mvp[2]/2.0
                   dv[id1]   = dv[id1] - dv_mvp
                   
                # Check the noreso aircraft. Nobody avoids noreso aircraft. 
                # But noreso aircraft will avoid other aircraft
                if asas.swnoreso:
                    if ac2 in asas.noresolst: # -> Then id1 does not avoid id2. 
                        dv[id1] = dv[id1] + dv_mvp
                
                # Check the resooff aircraft. These aircraft will not do resolutions.
                if asas.swresooff:
                    if ac1 in asas.resoofflst: # -> Then id1 does not do any resolutions
                        dv[id1] = 0.0                
    
    
    # Determine new speed and limit resolution direction for all aicraft-------     

    # Resolution vector for all aircraft, cartesian coordinates
    dv = np.transpose(dv)

    # The old speed vector for all aircraft, cartesian coordinates
    v = np.array([traf.gseast, traf.gsnorth, traf.vs])

    # The new speed vector, cartesian coordinates (dv = 0 for conflict free a/c)
    newv = dv+v
    
    
    # Limit resolution direction if required-----------------------------------
    
    # Compute new speed vector in polar coordinates based on desired resolution 
    if asas.swresohoriz: # horizontal resolutions
        if asas.swresospd and not asas.swresohdg: # SPD only
            newtrack = traf.trk
            newgs    = np.sqrt(np.square(newv[0,:])+np.square(newv[1,:]))
            newvs    = traf.vs           
        elif asas.swresohdg and not asas.swresospd: # HDG only
            newtrack = np.degrees(np.arctan2(newv[0,:],newv[1,:])) %360.0
            newgs    = traf.gs
            newvs    = traf.vs  
        else: # SPD + HDG
            newtrack = np.degrees(np.arctan2(newv[0,:],newv[1,:])) %360.0
            newgs    = np.sqrt(np.square(newv[0,:])+np.square(newv[1,:]))
            newvs    = traf.vs 
    elif asas.swresovert: # vertical resolutions
        newtrack = traf.trk
        newgs    = traf.gs
        newvs    = newv[2,:]       
    else: # horizontal + vertical
        newtrack = np.degrees(np.arctan2(newv[0,:],newv[1,:])) %360.0
        newgs    = np.sqrt(np.square(newv[0,:])+np.square(newv[1,:]))
        newvs    = newv[2,:]
        

    # Determine ASAS module commands for all aircraft--------------------------
    
    # Cap the velocities
    newgscapped = np.maximum(asas.vmin,np.minimum(asas.vmax,newgs))
    vscapped    = np.maximum(asas.vsmin,np.minimum(asas.vsmax,newvs))
    
    # Set ASAS module updates
    asas.trk = newtrack
    asas.spd = newgscapped
    asas.vs  = vscapped
    
    # To compute asas alt, timesolveV is used. timesolveV is a really big value (1e9)
    # when there is no conflict. Therefore asas alt is only updated when its 
    # value is less than the look-ahead time, because for those aircraft are in conflict
    altCondition           = np.logical_and(timesolveV<asas.dtlookahead, np.abs(dv[2,:])>0.0)
    asasalttemp            = asas.vs*timesolveV + traf.alt
    asas.alt[altCondition] = asasalttemp[altCondition] 
    
    # If resolutions are limited in the horizontal direction, then asasalt should
    # be equal to auto pilot alt (aalt). This is to prevent a new asasalt being computed 
    # using the auto pilot vertical speed (traf.avs) using the code in line 106 (asasalttemp) when only
    # horizontal resolutions are allowed.
    asas.alt = asas.alt*(1-asas.swresohoriz) + traf.selalt*asas.swresohoriz
    
    # NOTE:
    # WHEN PRIORITY IS ON, THE ABOVE LINES DON'T WORK WELL WHEN CLIMBING/DESCENDING
    # HAVE PRIORITY BECAUSE WE ARE UPDATING asas.ALT for all aircraft in conflict
    # THIS CAN BE FIXED BY CHECKING if dv FOR any aircraft is 0. FOR THOSE AIRCRAFT,
    # FOLLOW THE AUTOPILOT GUIDANCE. THIS IS ALSO THE OVERSHOOTING BUSINESS NEEDED FOR 
    # LAYERS -> CONDITION 2 IS REQUIRED BELOW!
    
           
#======================= Modified Voltage Potential ===========================
           

def MVP(traf, asas, id1, id2):
    """Modified Voltage Potential (MVP) resolution method"""
    
    
    # Preliminary calculations-------------------------------------------------
    
    # Get distance and qdr between id1 and id2
    dist = asas.dist[id1,id2]
    qdr  = asas.qdr[id1,id2]
    
    # Convert qdr from degrees to radians
    qdr = np.radians(qdr)
   
    # Relative position vector between id1 and id2
    drel = np.array([np.sin(qdr)*dist, \
                     np.cos(qdr)*dist, \
                     traf.alt[id2]-traf.alt[id1]])
       
    # Write velocities as vectors and find relative velocity vector              
    v1 = np.array([traf.gseast[id1], traf.gsnorth[id1], traf.vs[id1]])
    v2 = np.array([traf.gseast[id2], traf.gsnorth[id2], traf.vs[id2]])
    vrel = np.array(v2-v1) 
    
    
    # Horizontal resolution----------------------------------------------------
    
    # Get the time to solve conflict horizontally -> tcpa
    tcpa = asas.tcpa[id1,id2] 
    
    # Find horizontal distance at the tcpa (min horizontal distance)
    dcpa  = drel + vrel*tcpa
    dabsH = np.sqrt(dcpa[0]*dcpa[0]+dcpa[1]*dcpa[1])
        
    # Compute horizontal intrusion
    iH = asas.Rm - dabsH
    
    # Exception handlers for head-on conflicts
    # This is done to prevent division by zero in the next step
    if dabsH <= 10.:
        dabsH = 10.
        dcpa[0] = 10.
        dcpa[1] = 10.
        
    # Compute the resolution velocity vector in horizontal direction
    # abs(tcpa) because it bcomes negative during intrusion
    dv1 = (iH*dcpa[0])/(abs(tcpa)*dabsH)  
    dv2 = (iH*dcpa[1])/(abs(tcpa)*dabsH)
    
    # If intruder is outside the ownship PZ, then apply extra factor
    # to make sure that resolution does not graze IPZ
    if asas.Rm<dist and dabsH<dist:
        erratum=np.cos(np.arcsin(asas.Rm/dist)-np.arcsin(dabsH/dist))
        dv1 = dv1/erratum
        dv2 = dv2/erratum
        
    
    # Vertical resolution------------------------------------------------------
    
    # Compute the  vertical intrusion
    # Amount of vertical intrusion dependent on vertical relative velocity
    iV = asas.dhm if abs(vrel[2])>0.0 else asas.dhm-abs(drel[2])
    
    # Get the time to solve the conflict vertically - tsolveV
    tsolV = abs(drel[2]/vrel[2]) if abs(vrel[2])>0.0 else asas.tinconf[id1,id2]
    
    # If the time to solve the conflict vertically is longer than the look-ahead time,
    # because the the relative vertical speed is very small, then solve the intrusion
    # within tinconf
    if tsolV>asas.dtlookahead:
        tsolV = asas.tinconf[id1,id2]
        iV    = asas.dhm
    
    # Compute the resolution velocity vector in the vertical direction
    # The direction of the vertical resolution is such that the aircraft with
    # higher climb/decent rate reduces their climb/decent rate    
    dv3 = np.where(abs(vrel[2])>0.0,  (iV/tsolV)*(-vrel[2]/abs(vrel[2])), (iV/tsolV))
    
    # It is necessary to cap dv3 to prevent that a vertical conflict 
    # is solved in 1 timestep, leading to a vertical separation that is too 
    # high (high vs assumed in traf). If vertical dynamics are included to 
    # aircraft  model in traffic.py, the below three lines should be deleted.
#    mindv3 = -400*fpm# ~ 2.016 [m/s]
#    maxdv3 = 400*fpm
#    dv3 = np.maximum(mindv3,np.minimum(maxdv3,dv3))

    
    # Combine resolutions------------------------------------------------------

    # combine the dv components 
    dv = np.array([dv1,dv2,dv3])
    
    import pdb
    pdb.set_trace()
    
    return dv, tsolV
    
#============================= Priority Rules =================================    
    
def prioRules(traf, priocode, dv_mvp, dv1, dv2, id1, id2):
    ''' Apply the desired priority setting to the resolution '''
    
    # Primary Free Flight prio rules (no priority)
    if priocode == "FF1": 
        # since cooperative, the vertical resolution component can be halved, and then dv_mvp can be added
        dv_mvp[2] = dv_mvp[2]/2.0
        dv1 = dv1 - dv_mvp
        dv2 = dv2 + dv_mvp 
    
    # Secondary Free Flight (Cruising aircraft has priority, combined resolutions)    
    if priocode == "FF2": 
        # since cooperative, the vertical resolution component can be halved, and then dv_mvp can be added
        dv_mvp[2] = dv_mvp[2]/2.0
        # If aircraft 1 is cruising, and aircraft 2 is climbing/descending -> aircraft 2 solves conflict
        if abs(traf.vs[id1])<0.1 and abs(traf.vs[id2]) > 0.1:
            dv2 = dv2 + dv_mvp
        # If aircraft 2 is cruising, and aircraft 1 is climbing -> aircraft 1 solves conflict
        elif abs(traf.vs[id2])<0.1 and abs(traf.vs[id1]) > 0.1:
            dv1 = dv1 - dv_mvp
        else: # both are climbing/descending/cruising -> both aircraft solves the conflict
            dv1 = dv1 - dv_mvp
            dv2 = dv2 + dv_mvp
    
    # Tertiary Free Flight (Climbing/descending aircraft have priority and crusing solves with horizontal resolutions)          
    elif priocode == "FF3": 
        # If aircraft 1 is cruising, and aircraft 2 is climbing/descending -> aircraft 1 solves conflict horizontally
        if abs(traf.vs[id1])<0.1 and abs(traf.vs[id2]) > 0.1:
            dv_mvp[2] = 0.0
            dv1       = dv1 - dv_mvp
        # If aircraft 2 is cruising, and aircraft 1 is climbing -> aircraft 2 solves conflict horizontally
        elif abs(traf.vs[id2])<0.1 and abs(traf.vs[id1]) > 0.1:
            dv_mvp[2] = 0.0
            dv2       = dv2 + dv_mvp
        else: # both are climbing/descending/cruising -> both aircraft solves the conflict, combined
            dv_mvp[2] = dv_mvp[2]/2.0
            dv1       = dv1 - dv_mvp
            dv2       = dv2 + dv_mvp
            
    # Primary Layers (Cruising aircraft has priority and clmibing/descending solves. All conflicts solved horizontally)        
    elif priocode == "LAY1": 
        dv_mvp[2] = 0.0
        # If aircraft 1 is cruising, and aircraft 2 is climbing/descending -> aircraft 2 solves conflict horizontally
        if abs(traf.vs[id1])<0.1 and abs(traf.vs[id2]) > 0.1:
            dv2 = dv2 + dv_mvp
        # If aircraft 2 is cruising, and aircraft 1 is climbing -> aircraft 1 solves conflict horizontally
        elif abs(traf.vs[id2])<0.1 and abs(traf.vs[id1]) > 0.1:
            dv1 = dv1 - dv_mvp
        else: # both are climbing/descending/cruising -> both aircraft solves the conflict horizontally
            dv1 = dv1 - dv_mvp
            dv2 = dv2 + dv_mvp
    
    # Secondary Layers (Climbing/descending aircraft has priority and cruising solves. All conflicts solved horizontally)
    elif priocode ==  "LAY2": 
        dv_mvp[2] = 0.0
        # If aircraft 1 is cruising, and aircraft 2 is climbing/descending -> aircraft 1 solves conflict horizontally
        if abs(traf.vs[id1])<0.1 and abs(traf.vs[id2]) > 0.1:
            dv1 = dv1 - dv_mvp
        # If aircraft 2 is cruising, and aircraft 1 is climbing -> aircraft 2 solves conflict horizontally
        elif abs(traf.vs[id2])<0.1 and abs(traf.vs[id1]) > 0.1:
            dv2 = dv2 + dv_mvp
        else: # both are climbing/descending/cruising -> both aircraft solves the conflic horizontally
            dv1 = dv1 - dv_mvp
            dv2 = dv2 + dv_mvp
       
    return dv1, dv2
    