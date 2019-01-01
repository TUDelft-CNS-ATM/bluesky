"""
Airspace Design Contest Scenario Generator:

Draw circles and runways in use
Generate traffic at edges and at airports
Make sure scenario can be saved

"""


from bluesky import stack,traf,sim,tools  #, settings, navdb, traf, sim, scr, tools
from trafgenclasses import Source, setcircle
from bluesky.tools.position import txt2pos
#from bluesky.tools import areafilter
#from bluesky.tools.aero import vtas2cas,ft
#from bluesky.tools.misc import degto180


import numpy as np

# Default values
ctrlat = 52.6
ctrlon = 5.4
radius = 230.


def init_plugin():
    print("Initialising contest scenario generator")

    # Create an empty geovector list
    reset()

    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name'      : 'TRAFGEN',
        'plugin_type'      : 'sim',
        'update_interval'  :  0.1,

        # The update function is called after traffic is updated.
        'update':          update,

        # The preupdate function is called before traffic is updated. Use this
        #'preupdate':       preupdate,

        # Reset contest
        'reset':         reset
        }

    # Add two commands: GEOVECTOR to define a geovector for an area
    stackfunctions = {
        # Starting contest traffic generation
        'TRAFGEN': [
            'TRAFGEN [location],cmd,[arg, arg, ...]',
            'string',
            trafgencmd,
            'CONTEST command']
        }

    # init_plugin() should always return these two dicts.

    return config, stackfunctions

def reset():
    # Contest global variables
    global ctrlat,ctrlon,radius,dtsegment,drains,sources,rwsdep,rwsarr

    # Set default parameters for spawning circle

    swcircle = False
    ctrlat = 52.6  # [deg]
    ctrlon = 5.4  # [deg]
    radius = 230.0  # [nm]

    # Draw circle
    stack.stack("CIRCLE SPAWN," + str(ctrlat) + "," + str(ctrlon) + "," + str(radius))

    # Average generation interval in [s] per segment
    dtsegment = 12 * [1.0]

    # drains: dictionary of drains
    sources     = dict([])

    return


### Periodic update functions that are called by the simulation. You can replace
### this by anything, so long as you communicate this in init_plugin

#def preupdate(): # To be safe preupdate is used iso update
#    pass
#    return


def update(): # Update all sources and drain
    for src in sources:
        sources[src].update()

    return

### Other functions of your plug-in
def trafgencmd(cmdline):
    global ctrlat,ctrlon,radius,dtsegment,drains,sources,rwsdep,rwsarr

    cmd,args = splitline(cmdline)

    #print("TRAFGEN: cmd,args=",cmd,args)

    if cmd=="CIRCLE" or cmd=="CIRC":


        # Draw circle
        try:
            swcircle = True
            ctrlat = float(args[0])
            ctrlon = float(args[1])
            radius = float(args[2])
            setcircle(ctrlat, ctrlon, radius)
        except:
            return False,'TRAFGEN ERROR while reading CIRCLE command arguments (lat,lon,radius):'+str(args)
        stack.stack("DEL SPAWN")
        stack.stack("CIRCLE SPAWN," + str(ctrlat) + "," + str(ctrlon) + "," + str(radius))



    elif cmd=="SRC": # Define streams by source, give destinations
        name = args[0].upper()
        cmd = args[1].upper()
        cmdargs = args[2:]
        if name not in sources:
            success,posobj = txt2pos(name,ctrlat,ctrlon)
            if success:
                 sources[name] = Source(name,cmd,cmdargs)
        else:
            success = True

        if success:
            if cmd=="RUNWAY" or cmd=="RWY":
                sources[name].addrunways(cmdargs)
            elif cmd=="DEST":
                sources[name].adddest(cmdargs)
            elif cmd=="FLOW":
                sources[name].setflow(cmdargs[0])
            elif cmd=="TYPES" or cmd=="TYPE":
                sources[name].addactypes(cmdargs)
        else:
            return False,"TRAFGEN SRC ERROR "+name+" NOT FOUND"

    return True

def splitline(rawline):
    # Interpet string like a command with arguments
    # Replace multiple spaces by one space

    line = rawline.strip().upper()
    if line.count("#")>=1:
        icomment = line.index("#")
        line  = line[:icomment]

    while line.count("  ") > 0:
        line = line.replace("  ", " ")

    # and remove spaces around commas
    while line.count(", ") > 0:
        line = line.replace(", ", ",")

    while line.count(" ,") > 0:
        line = line.replace(" ,", ",")

    # Replace remaining spaces, which are separators, by comma
    line = line.strip().replace(" ",",")

    # Split using commas
    args = line.split(",")
    if len(args)>=1:
        cmd = args[0]
    else:
        cmd = ""
    return cmd,args[1:]

