""" Geovectoring: for an area define an allowed interval for each component of the 3D speed vector (gs,trk,vs):

first argument is the are on whuch the geovector shoudl apply

Geovector is defined as:
 (  [ gsmin,gsmax ]  ) [kts]
 (  [trkmin,trkmax]  ) [deg]
 (  [ vsmin,vsmax ]  ) [fpm]


 """


from bluesky import stack,traf,sim,tools  #, settings, navdb, traf, sim, scr, tools
from bluesky.tools import areafilter
from bluesky.tools.aero import tas2cas
from bluesky.tools.misc import degto180

import numpy as np

def init_plugin():

    # Create an empty geovector list
    reset()

    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'GEOVECTOR',
        'plugin_type':     'sim',
        'update_interval': 1.0,

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
            'GEOVECTOR area,[gsmin,gsmax,trkmin,trkmax,vsmin,vsmax]/RESET',
            'txt,[speed,speed,hdg,hdg,vspd,vspd]',
            defgeovec,
            'Define a geovector for an area defined with the AREA command']
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
    # Apply each geovector
    for vec in geovecs:
        areaname = vec[0]
        if areafilter.hasarea(areaname):
            swinside = checkInside(areaname, traf.lat, traf.lon, traf.alt)
            swoutside = np
            gsmin,gsmax,trkmin,trkmax,vsmin,vsmax = vec[1:]

            # -----Ground speed limiting
            # For now assume no wind:  so use tas as gs
            if gsmin:
                casmin = tas2cas(np.ones(traf.ntraf)*gsmin,traf.alt)
                bs.traf.selspd = np.logical_not(swinside)*bs.traf.selspd +  \
                        swinside*np.maximum(casmin,bs.traf.selspd)
            if gsmax:
                casmax = tas2cas(np.ones(traf.ntraf)*gsmax,traf.alt)
                bs.traf.selspd = np.logical_not(swinside) * bs.traf.selspd +  \
                        swinside * np.minimum(casmax, bs.traf.selspd)


            #------ Limit Track(so hdg)
            # Max track interval is 180 degrees to avoid ambiguity of what is inside the interval

            if trkmin and trkmax:
                # Use degto180 to avodi problems for e.g interval [350,30]
                usemin = swinside&(degto180(bs.traf.trk - trkmin)<0) # Left of minimum
                usemax = swinside&(degto180(bs.traf.trk - trkmax)>0) # Right of maximum
                usetrk = np.logical_not(usemin&usemax)    # None of the above: use orig trk

                bs.traf.ap.trk = usetrk*bs.traf.ap.trk + usemin*trkmin + usemax*trkmax


            # -----Ground speed limiting
            # For now assume no wind:  so use tas as gs
            if vsmin:
                bs.traf.selvs[swinside&(bs.traf.vs<vsmin)] = vsmin

            if vsmax:
                bs.traf.selvs[swinside & (bs.traf.vs > vsmax)] = vsmax


    return

def update(): # Not used
    return

def reset():
    geovecs = [] # [[area,gsmin,gsmax,trkin,trkmax,vsmin,vsmax]]
    return

### Other functions of your plug-in

def defgeovec(area="", gsmin=None, gsmax=None, trkmin=None, trkmax=None, vsmin=None, vsmax=None):

    # We need an area to do anything
    if area=="":
        return False

    # Remove old geovector for this area
    i = 0
    while i<len(geovecs):
        if geovecs[i][0].upper()==area.upper():
            del geovecs[i]
        else:
            i = i + 1

    # Only add it if an interval is given
    if gsmin or gsmax or (trkmin and trkmax) or vsmin or vsmax:
        geovecs.append(area,gsmin,gsmax,trkmin,trkmax,vsmin,vsmax)

    return True, ""

def delgeovec(area=""):
    return defgeovec(area) # Using all Nones will delete the geovector
