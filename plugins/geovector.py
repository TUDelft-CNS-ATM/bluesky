""" Geovectoring: for an area define an allowed interval for each component of the 3D speed vector (gs,trk,vs):

first argument is the are on whuch the geovector shoudl apply

Geovector is defined as:
 (  [ gsmin,gsmax ]  ) [kts]
 (  [trkmin,trkmax]  ) [deg]
 (  [ vsmin,vsmax ]  ) [fpm]


 """
import numpy as np

from bluesky import traf  #, settings, navdb, traf, sim, scr, tools
from bluesky.tools import areafilter
from bluesky.tools.aero import vtas2cas,ft
from bluesky.tools.misc import degto180


# A dictionary of areas with a geovector specification
geovecs = dict()


class GeoVector:
    ''' Geovector specification class. '''
    def __init__(self, areaname, gsmin, gsmax,
                 trkmin, trkmax,
                 vsmin, vsmax):
        self.areaname = areaname
        self.gsmin = gsmin
        self.gsmax = gsmax
        self.trkmin = trkmin
        self.trkmax = trkmax
        self.vsmin = vsmin
        self.vsmax = vsmax
        self.previnside = set()
        self.prevstatus = dict()

    def __str__(self):
        ''' Pretty printed geovector information. '''
        return f'gs: {self.gsmin}-{self.gsmax} [m/s], ' + \
            f'{self.trkmin}-{self.trkmax} trk[deg], ' + \
                f'{self.vsmin}-{self.vsmax} vs[m/s]'


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
    for areaname, vec in geovecs.items():
        if areafilter.hasArea(areaname):
            swinside  = areafilter.checkInside(areaname, traf.lat, traf.lon, traf.alt)

            insids = set(np.array(traf.id)[swinside])
            newids = insids - vec.previnside
            delids = vec.previnside - insids
            # Store LNAV/VNAV status of new aircraft
            for acid in newids:
                idx = traf.id2idx(acid)
                vec.prevstatus[acid] = [traf.swlnav[idx], traf.swvnav[idx]]

            # Revert aircraft who have exited the geovectored area to their original status
            for acid in delids:
                idx = traf.id2idx(acid)
                if idx >= 0:
                    traf.swlnav[idx], traf.swvnav[idx] = vec.prevstatus.pop(acid)
            
            vec.previnside = insids
            # -----Ground speed limiting
            # For now assume no wind:  so use tas as gs
            if vec.gsmin is not None:
                casmin = vtas2cas(np.ones(traf.ntraf) * vec.gsmin, traf.alt)
                usemin = traf.selspd < casmin
                traf.selspd[swinside & usemin] = casmin[swinside & usemin]
                traf.swvnav[swinside & usemin] = False

            if vec.gsmax is not None:
                casmax = vtas2cas(np.ones(traf.ntraf) * vec.gsmax, traf.alt)
                usemax = traf.selspd > casmax
                traf.selspd[swinside & usemax] = casmax[swinside & usemax]
                traf.swvnav[swinside & usemax] = False

            #------ Limit Track(so hdg)
            # Max track interval is 180 degrees to avoid ambiguity of what is inside the interval

            if None not in [vec.trkmin, vec.trkmax]:
                # Use degto180 to avodi problems for e.g interval [350,30]
                usemin = swinside & (degto180(traf.trk - vec.trkmin) < 0) # Left of minimum
                usemax = swinside & (degto180(traf.trk - vec.trkmax) > 0) # Right of maximum

                #print(usemin,usemax)
                traf.swlnav[swinside & (usemin | usemax)] = False

                traf.ap.trk[swinside & usemin] = vec.trkmin
                traf.ap.trk[swinside & usemax] = vec.trkmax

            # -----Ground speed limiting
            # For now assume no wind:  so use tas as gs
            if vec.vsmin is not None:
                traf.selvs[swinside & (traf.vs < vec.vsmin)] = vec.vsmin
                traf.swvnav[swinside & (traf.vs < vec.vsmin)] = False
                # Activate V/S mode by using a slightly higher altitude than current values
                traf.selalt[swinside & (traf.vs < vec.vsmin)] = traf.alt[swinside & (traf.vs < vec.vsmin)] + \
                                                            np.sign(vec.vsmin)*200.*ft

            if vec.vsmax is not None:
                traf.selvs[swinside & (traf.vs > vec.vsmax)] = vec.vsmax
                traf.swvnav[swinside & (traf.vs < vec.vsmax)] = False
                # Activate V/S mode by using a slightly higher altitude than current values
                traf.selalt[swinside & (traf.vs > vec.vsmax)] = traf.alt[swinside & (traf.vs > vec.vsmax)] + \
                                                            np.sign(vec.vsmax)*200.*ft

    return

def update(): # Not used
    return

def reset():
    geovecs.clear() # [[area,gsmin,gsmax,trkin,trkmax,vsmin,vsmax]]
    return

### Other functions of your plug-in

def defgeovec(areaname="", spdmin=None, spdmax=None, trkmin=None, trkmax=None, vspdmin=None, vspdmax=None):
    #print ("defgeovec input=",areaname,gsmin,gsmax,trkmin,trkmax,vspdmin,vspdmax,sep="|")

    # We need an area to do anything
    if not areaname:
        return False, "We need an area"

    if not (spdmin or spdmax or (trkmin and trkmax) or vspdmin or vspdmax):
        if areaname in geovecs:
            return True, areaname + " uses " + str(geovecs[areaname])
        return False, "No geovector found for " + areaname

    # Remove old geovector for this area (if it exists)
    geovecs.pop(areaname, None)

    # Only add it if an interval is given
    if spdmin or spdmax or (trkmin and trkmax) or vspdmin or vspdmax:
        # Allow wrong order of min and max (note for hdg, as 355,10 is valid range!)
        if None not in [spdmin, spdmax]:
            gsmin = min(spdmin,spdmax)
            gsmax = max(spdmin,spdmax)
        else:
            gsmin = spdmin
            gsmax = spdmax

        if None not in [vspdmin, vspdmax]:
            vsmin = min(vspdmin,vspdmax)
            vsmax = max(vspdmin,vspdmax)
        else:
            vsmin = vspdmin
            vsmax = vspdmax

        # Add geovector to the dict of geovectors
        geovecs[areaname] = GeoVector(areaname,
                        gsmin, gsmax,
                        trkmin,trkmax,
                        vsmin, vsmax)

    return True

def delgeovec(area=""):
    # Delete geovector specification for area if it exists
    found = geovecs.pop(area)
    if not found:
        return False, "No geovector found for " + area

    return True, 'Deleted geovector specification for ' + area
