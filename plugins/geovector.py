""" Geovectoring: for an area define an allowed interval for each component of the 3D speed vector (gs,trk,vs):

first argument is the are on whuch the geovector shoudl apply

Geovector is defined as:
 (  [ gsmin,gsmax ]  ) [kts]
 (  [trkmin,trkmax]  ) [deg]
 (  [ vsmin,vsmax ]  ) [fpm]


 """


from bluesky import stack,traf,sim,tools  #, settings, navdb, traf, sim, scr, tools
from bluesky.tools import areafilter
from bluesky.tools.aero import vtas2cas,ft
from bluesky.tools.misc import degto180


import numpy as np

def init_plugin():

    # Create an empty geovector list
    reset()

    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name'      : 'GEOVECTOR',
        'plugin_type'      : 'sim',
        'update_interval'  :  1.0,

        # The update function is called after traffic is updated. Use this if you
        # want to do things as a result of what happens in traffic. If you need to
        # something before traffic is updated please use preupdate.
        'update':          update,

        # The preupdate function is called before traffic is updated. Use this
        # function to provide settings that need to be used by traffic in the current
        # timestep. Examples are ASAS, which can give autopilot commands to resolve
        # a conflict.
        'preupdate':       preupdate,

        # Reset all geovectors
        'reset':         reset
        }

    # Add two commands: GEOVECTOR to define a geovector for an area
    stackfunctions = {
        # Defining a geovector
        'GEOVECTOR': [
            'GEOVECTOR area,[gsmin,gsmax,trkmin,trkmax,vsmin,vsmax]',
            'txt,[spd,spd,hdg,hdg,vspd,vspd]',
            defgeovec,
            'Define a geovector for an area defined with the BOX,POLY(ALT) area commands']
        ,
        # Delete a geovector (same effect as using a geovector without  any values
        'DELGEOVECTOR': [
            'DELGEOVECTOR area',
            'txt',
            delgeovec,
            'Remove geovector from the area ']
        }
    # init_plugin() should always return these two dicts.
    return config, stackfunctions


### Periodic update functions that are called by the simulation. You can replace
### this by anything, so long as you communicate this in init_plugin

def preupdate(): # To be safe preupdate is used iso update
    applygeovec()
    return

def applygeovec():
    # Apply each geovector
    for vec in geovecs:
        areaname = vec[0]
        if areafilter.hasArea(areaname):
            swinside  = areafilter.checkInside(areaname, traf.lat, traf.lon, traf.alt)

            gsmin,gsmax,trkmin,trkmax,vsmin,vsmax = vec[1:]

            # -----Ground speed limiting
            # For now assume no wind:  so use tas as gs
            if gsmin:
                casmin = vtas2cas(np.ones(traf.ntraf)*gsmin,traf.alt)
                usemin = traf.selspd<casmin
                traf.selspd[swinside & usemin] = casmin[swinside & usemin]

            if gsmax:
                casmax = vtas2cas(np.ones(traf.ntraf)*gsmax,traf.alt)
                usemax = traf.selspd > casmax
                traf.selspd[swinside & usemax] = casmax[swinside & usemax]

            #------ Limit Track(so hdg)
            # Max track interval is 180 degrees to avoid ambiguity of what is inside the interval

            if trkmin and trkmax:

                # Use degto180 to avodi problems for e.g interval [350,30]
                usemin = swinside & (degto180(traf.trk - trkmin)<0) # Left of minimum
                usemax = swinside & (degto180(traf.trk - trkmax)>0) # Right of maximum

                #print(usemin,usemax)

                traf.ap.trk[swinside & usemin] = trkmin
                traf.ap.trk[swinside & usemax] = trkmax

            # -----Ground speed limiting
            # For now assume no wind:  so use tas as gs
            if vsmin:

                traf.selvs[swinside & (traf.vs<vsmin)] = vsmin

                # Activate V/S mode by using a slightly higher altitude than current values
                traf.selalt[swinside & (traf.vs < vsmin)] = traf.alt[swinside & (traf.vs < vsmin)] + \
                                                            np.sign(vsmin)*200.*ft

            if vsmax:
                traf.selvs[swinside & (traf.vs > vsmax)] = vsmax

                # Activate V/S mode by using a slightly higher altitude than current values
                traf.selalt[swinside & (traf.vs > vsmax)] = traf.alt[swinside & (traf.vs > vsmax)] + \
                                                            np.sign(vsmax)*200.*ft

    return

def update(): # Not used
    return

def reset():
    global geovecs
    geovecs = [] # [[area,gsmin,gsmax,trkin,trkmax,vsmin,vsmax]]
    return

### Other functions of your plug-in

def defgeovec(area="", spdmin=None, spdmax=None, trkmin=None, trkmax=None, vspdmin=None, vspdmax=None):
    global geovecs
    #print ("defgeovec input=",area,gsmin,gsmax,trkmin,trkmax,vspdmin,vspdmax,sep="|")

    # We need an area to do anything
    if area=="":
        return False,"We need an area"

    if not (spdmin or spdmax or (trkmin and trkmax) or vspdmin or vspdmax):
        i = 0
        idx = -1
        while i < len(geovecs):
            if geovecs[i][0].upper() == area.upper():
                idx = i
                break

            i = i + 1
        if idx>=0:
            return True,area+" uses "+str(geovecs[idx][1:])+" gs[m/s], trk[deg], vs[m/s]"
        else:
            return False,"No geovector found for "+area


    # Remove old geovector for this area
    i = 0
    while i<len(geovecs):
        if geovecs[i][0].upper()==area.upper():
            del geovecs[i]
        else:
            i = i + 1

    # Only add it if an interval is given
    if spdmin or spdmax or (trkmin and trkmax) or vspdmin or vspdmax:
        # Allow wrong order of min and max (note for hdg, as 355,10 is valid range!)
        if spdmin and spdmax:
            gsmin = min(spdmin,spdmax)
            gsmax = max(spdmin,spdmax)
        else:
            gsmin = spdmin
            gsmax = spdmax

        if vspdmin and vspdmax:
            vsmin = min(vspdmin,vspdmax)
            vsmax = max(vspdmin,vspdmax)
        else:
            vsmin = vspdmin
            vsmax = vspdmax

        # Add geovector to the list of geovectors
        geovecs.append([area,
                        gsmin, gsmax,
                        trkmin,trkmax,
                        vsmin, vsmax ])

    return True

def delgeovec(area=""):
    found = False
    i = 0
    while i<len(geovecs):
        if geovecs[i][0].upper()==area.upper():
            found = True
            del geovecs[i]
        else:
            i = i + 1

    if not found:
        return False,"No geovector found for "+area
    else:
        return True
