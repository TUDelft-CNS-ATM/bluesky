# -*- coding: utf-8 -*-
"""
Created on Wed Mar 04 14:27:44 2015

@author: Jerom Maas
"""

import numpy as np
from aero import nm, ft
import MVP

def start(dbconf):
    dbconf.CRname="Swarm"
    dbconf.Rswarm= 7.5*nm #[m]
    dbconf.dhswarm = 1500*ft  #[m]
    
    dbconf.Swarmweights=np.array([10,3,1])
    pass

def resolve(dbconf):
    tr=dbconf.traf
    
    # Find matrix of neighbouring aircraft withing swarm distance
    dx=dbconf.dx
    dy=dbconf.dy-np.eye(tr.ntraf)*1e9 #setting distance of A/C to itself to 0,\
                                    #correcting the distance from CASAS line 109
    
    close=np.logical_and(dx**2+dy**2<dbconf.Rswarm**2,\
                    np.abs(dbconf.dalt)     <dbconf.dhswarm)
                    
    trkdif=tr.trk.reshape(1,tr.ntraf)-tr.trk.reshape(tr.ntraf,1)
    dtrk=(trkdif+180)%360-180
    samedirection=np.abs(dtrk)<90
    
    selected=np.logical_and(close,samedirection)
    own=np.eye(tr.ntraf,dtype='bool')
    
    Swarming=np.logical_or(selected,own)
    
    # First do conflict resolution following MVP
    MVP.resolve(dbconf)    
    
    # Find desired speed vector after Collision Avoidance or Autopilot 
    ca_trk = tr.asasactive*tr.asashdg+(1-tr.asasactive)*tr.ahdg
    ca_cas = tr.asasactive*tr.asasspd+(1-tr.asasactive)*tr.aspd
    ca_vs = tr.asasactive*tr.asasvsp+(1-tr.asasactive)*tr.avs
    
    # Add factor of Velocity Alignment to speed vector
    hspeed=np.ones((tr.ntraf,tr.ntraf))*tr.cas
    va_cas=np.average(hspeed,axis=1,weights=Swarming)
    
    vspeed=np.ones((tr.ntraf,tr.ntraf))*tr.vs
    va_vs=np.average(vspeed,axis=1,weights=Swarming)
    
    avgdtrk=np.average(dtrk,axis=1,weights=Swarming)    
    va_trk = tr.trk+avgdtrk
    
    # Add factor of Flock Centering to speed vector
    dxflock=dx+np.eye(tr.ntraf)*dbconf.u/100.
    dyflock=dy+np.eye(tr.ntraf)*dbconf.v/100.
    
    fc_dx=np.average(dxflock,axis=1,weights=Swarming)
    fc_dy=np.average(dyflock,axis=1,weights=Swarming)
    
    z=np.ones((tr.ntraf,tr.ntraf))*tr.alt
    fc_dz=np.average(z,axis=1,weights=Swarming)-tr.alt
    
    fc_trk=np.degrees(np.arctan2(fc_dx,fc_dy))
    fc_cas=tr.cas
    ttoreach=np.sqrt(fc_dx**2+fc_dy**2)/fc_cas
    fc_vs=np.where(ttoreach==0,0,fc_dz/ttoreach)
    
    # Find final Swarming directions
    trks=np.array([ca_trk,va_trk,fc_trk])
    cass=np.array([ca_cas,va_cas,fc_cas])
    vss=np.array([ca_vs,va_vs,fc_vs])
    
    trksrad=np.radians(trks)
    vxs=cass*np.sin(trksrad)
    vys=cass*np.cos(trksrad)
    
    Swarmvx=np.average(vxs,axis=0,weights=dbconf.Swarmweights)
    Swarmvy=np.average(vys,axis=0,weights=dbconf.Swarmweights)
    Swarmhdg=np.degrees(np.arctan2(Swarmvx,Swarmvy))
    Swarmcas=np.average(cass,axis=0,weights=dbconf.Swarmweights)
    Swarmvs=np.average(vss,axis=0,weights=dbconf.Swarmweights)

    # Cap the velocity
    Swarmcascapped=np.maximum(dbconf.vmin,np.minimum(dbconf.vmax,Swarmcas))    
    # Assign Final Swarming directions to traffic
    tr.asashdg=Swarmhdg
    tr.asasspd=Swarmcascapped
    tr.asasvsp =Swarmvs
    tr.asasalt=np.sign(Swarmvs)*1e5
    
    # Make sure that all aircraft follow these directions
    tr.asasactive.fill(True)
    pass
