# -*- coding: utf-8 -*-
"""
Created on Wed Mar 04 14:27:44 2015

@author: Jerom Maas
"""
import os
import numpy as np
import Difgamelearnerfunctions as dg
from bluesky.tools.aero import nm, kts, vtas2eas


def start(dbconf):
    filepath=os.path.abspath("Difgame.py")
    fpath=filepath[:-10]+"\CDRmethods\DifgameActions.npy"
    dbconf.Controls = np.load(fpath)

    # Discretization numbers: in how many elements is discretized
    dbconf.dn_state=[9,9,5,5,8] #states
    dn_action=[3,3,3,3]  #actions
    dn= dbconf.dn_state + dn_action

    gridsize=5*nm

    # State variables
    dbconf.xw=dg.ndisc(0,gridsize,dn[0])        # [m]
    dbconf.yw=dg.ndisc(12.5*nm,gridsize,dn[1])  # [m]
    dbconf.v_o=dg.ndisc(140,15,dn[2])           # [m/s]
    dbconf.v_w=dg.ndisc(140,15,dn[3])           # [m/s]
    dbconf.phi=dg.cdisc(-1,1,dn[4])*np.pi       # [radians]

    Discstates=[dbconf.xw,dbconf.yw,dbconf.v_o,dbconf.v_w,dbconf.phi]
    dbconf.statebins=dg.discbins(Discstates)

    # Control variables
    accmax=1.*kts                               # [m/s]
    bankmax=np.radians(25)                      # [radians]

    dbconf.a_o=dg.bdisc(-1,1,dn[5])*accmax      # [m/s]
    dbconf.a_w=dg.bdisc(-1,1,dn[6])*accmax      # [m/s]
    dbconf.b_o=dg.bdisc(-1,1,dn[7])*bankmax     # [radians]
    dbconf.b_w=dg.bdisc(-1,1,dn[8])*bankmax     # [radians]

    dbconf.dist_a = 10 * nm                     #[m]
    dbconf.dist_b = 12.5 * nm                   #[m]

    # Make list of considered states
    sshape=(len(dbconf.xw),len(dbconf.yw),len(dbconf.v_o),len(dbconf.v_w),len(dbconf.phi))
    dbconf.conflictstate=dg.consideredstates(sshape,Discstates)

def resolve(dbconf, traf):
    # -------------------------------------------------------------------------
    # First, perform special Conflict Detection to find which conflicts to resolve
    # Relative horizontal positions: p[a,b] is p from a to b
    qdrrel = np.radians(dbconf.qdr-traf.trk.reshape((len(traf.trk),1)))
    xrel= np.sin(qdrrel)*dbconf.dist
    yrel= np.cos(qdrrel)*dbconf.dist

    # Define the two borders:
    b1=np.where( yrel -np.abs(xrel) > -dbconf.dist_a                ,True,False)
    b2=np.where( xrel**2 + (yrel-dbconf.dist_b)**2 < (dbconf.dist_a+dbconf.dist_b)**2   ,True,False)

    # Create conflict matrix
    dbconf.swconfl_dg = b1*b2*(1.-np.eye(traf.ntraf))

    # -------------------------------------------------------------------------
    # Second, create a list of conflicts  and aircraft indices to iterate over

    dgconfidxs = np.where(dbconf.swconfl_dg)
    dgnconf = len(dgconfidxs[0])
    dgiown = dgconfidxs[0]
    dgioth = dgconfidxs[1]

    # -------------------------------------------------------------------------
    # Third, create control matrix and find controls
    # Also, aircraft in DGconflict should follow ASAS
    dbconf.asasactive.fill(False)
    controls=np.zeros((traf.ntraf,3),dtype=np.int)

    for i in range(dgnconf):
        id1=dgiown[i]
        id2=dgioth[i]
        dbconf.asasactive[id1]=True
        dbconf.asasactive[id2]=True

        ctrl=Difgamehor(traf, dbconf,id1,id2)
        controls[id1]+=ctrl-np.array([1,1,1])

    # Now, write controls as limits in traf class
    # First: limit superpositioned controls to maximum values
    controls+=np.array([1,1,1])
    maxcontrols=np.array([2,2,2])
    controls=np.where(controls<0,0,controls)
    controls=np.where(controls>maxcontrols,maxcontrols,controls)
    # From indices to values
    acccontrol=dbconf.a_o[controls[:,0]]
    bankcontrol=dbconf.b_o[controls[:,1]]
    climbcontrol=traf.avsdef*np.sign(controls[:,2])

    # Now assign in the traf class --------------------------------------------
    # Change autopilot desired speed
    dbconf.asasspd=vtas2eas(np.where(acccontrol==0,traf.gs,\
        np.where(acccontrol>0,traf.perf.vmax,traf.perf.vmin)),traf.alt)
    # Set acceleration for aircraft in conflict
    traf.ax=np.where(dbconf.asasactive,abs(acccontrol),kts)
    # Change autopilot desired heading
    dbconf.asashdg=np.where(bankcontrol==0,traf.trk,\
        np.where(bankcontrol>0,traf.trk+90,traf.trk-90))%360

    # Set bank angle for aircraft in conflict
    traf.ap.aphi=np.radians(np.where(dbconf.asasactive,np.abs(bankcontrol),25.))

    # Change autopilot desired altitude
    traf.aalt=np.where(climbcontrol==0,traf.alt,\
        np.where(climbcontrol>0,1e9,-1e9))
    # Set climb rate for aircraft in conflict
    dbconf.asasvsp=np.abs(climbcontrol)



def Difgamehor(traf, dbconf,own,wrn):
    # Does the ownship try to hit the other ship?
    pirates = (traf.id[own]=="WRN")

    if pirates: #switch to perspective of ownship, as dbconf.Control is in that form
        a=own
        own=wrn
        wrn=a

    #First, compute the five states
    dist=dbconf.dist[own,wrn]
    qdr=dbconf.qdr[own,wrn]
    phi=np.radians((traf.trk[wrn]-traf.trk[own]+180)%360-180)
    qdrrel=qdr-traf.trk[own]
    x=np.sin(np.radians(qdrrel))*dist
    y=np.cos(np.radians(qdrrel))*dist
    v_o=traf.gs[own]
    v_w=traf.adsbgs[wrn]

    state=np.array([x,y,v_o,v_w,phi])
    dstate=np.array([],dtype=np.int_)

    for p in range(5):
        # Construct the new discretized state
        whichbin=int(np.digitize([state[p]],dbconf.statebins[p])[0]-1)
        #if p in [3,4]: #if the state variable is an angle
        if p in [4]:    #if the state variable is an angle
            whichbin = whichbin % dbconf.dn_state[p]
        dstate=np.append(dstate,whichbin)

    # Find the correct horizontal controls: if 'pirates', look at wrongship
    horC = dbconf.Controls[tuple(dstate)][pirates]

    # Find vertical controls
    verticalconflict = np.abs(traf.alt[own]-traf.alt[wrn])<dbconf.R * self.resofach
    if verticalconflict:
        if traf.alt[own]<traf.alt[wrn]:
            verC = np.array([0])
        else:
            verC=np.array([2])
    else:
        verC = np.array([1])

    #Combine Horizontal and Vertical Controls
    C=np.append(horC,verC)

    return C
