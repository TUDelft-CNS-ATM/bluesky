"""
Airspace Design Contest Scenario Generator:

Draw circles and runways in use
Generate traffic at edges and at airports
Make sure scenario can be saved

"""


from bluesky import stack,traf,sim,tools,navdb  #, settings, navdb, traf, sim, scr, tools
from bluesky.plugins.trafgenclasses import Source, Drain, setcircle
from bluesky.tools.position import txt2pos
from bluesky.tools.geo import kwikpos
#from bluesky.tools import areafilter
#from bluesky.tools.aero import vtas2cas,ft
#from bluesky.tools.misc import degto180


import numpy as np

# Default values
swcircle   = False # No active circle
ctrlat     = 52.6  # Default values to resolve navdb names
ctrlon     = 5.4
radius     = 230.
globalgain = 1.0


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
    drains      = dict([])

    return


### Periodic update functions that are called by the simulation. You can replace
### this by anything, so long as you communicate this in init_plugin

#def preupdate(): # To be safe preupdate is used iso update
#    pass
#    return


def update(): # Update all sources and drain
    for src in sources:
        sources[src].update(globalgain)

    for drn in drains:
        drains[drn].update(globalgain)

    return

### Other functions of your plug-in
def trafgencmd(cmdline):
    global ctrlat,ctrlon,radius,dtsegment,drains,sources,rwsdep,rwsarr,globalgain

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

    elif cmd=="GAIN" or cmd=="FACTOR":
        # Set global gain on traffic density
        try:
            globalgain = float(args[0])
        except:
            return False, "TRAFGEN GAIN error: invalid value "+args[0]


    elif cmd=="SRC" or cmd == "SOURCE": # Define streams by source, give destinations
        name = args[0].upper()
        cmd = args[1].upper()
        cmdargs = args[2:]
        if name not in sources:
            if not name[:4] == "SEGM":
                success,posobj = txt2pos(name,ctrlat,ctrlon)
                if success:
                    aptlat, aptlon = posobj.lat, posobj.lon
                    sources[name] = Source(name,cmd,cmdargs)
                    if posobj.type == "rwy" and name.count("/")==1:
                        aptname, rwyname = name.split('/')
                        sources[name].polys += drawrwy(aptname,[rwyname],posobj.lat,posobj.lon,drawdeprwy)

            else:
                try:
                    brg = int(name[4:])
                    aptlat,aptlon = kwikpos(ctrlat,ctrlon,brg,radius)
                    sources[name] = Source(name, cmd, cmdargs)
                    success = True
                except:
                    success = False
        else:
            aptlat,aptlon = sources[name].lat,sources[name].lat
            success = True

        if success:
            if cmd[:6]=="RUNWAY" or cmd=="RWY":
                sources[name].setrunways(cmdargs)
                for polyname in sources[name].polys:
                    stack.stack("DEL "+polyname)
                sources[name].polys += drawrwy(name,cmdargs,aptlat,aptlon,drawdeprwy)

            elif cmd=="DEST":
                success = sources[name].adddest(cmdargs)
            elif cmd=="FLOW":
                success = sources[name].setflow(cmdargs[0])
            elif cmd=="TYPES" or cmd=="TYPE":
                sources[name].addactypes(cmdargs)
            elif cmd=="ALT": # Set fixed alt or interval
                sources[name].setalt(cmdargs)
            elif cmd=="SPD":
                sources[name].setspd(cmdargs)
            elif cmd=="HDG":
                sources[name].sethdg(cmdargs)
            if not success:
                return False, "TRAFGEN SRC ERROR"+cmd+" ".join(cmdargs)
        else:
            return False,"TRAFGEN SRC ERROR "+name+" NOT FOUND"

    elif cmd=="DRN" or cmd=="DRAIN":
        name = args[0].upper()
        cmd = args[1].upper()
        cmdargs = args[2:]
        if name in drains:
            aptlat, aptlon = drains[name].lat, drains[name].lat
            success = True
        else:
            # name not in drains: New drain defined
            if not name[:4]=="SEGM":
                success, posobj = txt2pos(name, ctrlat, ctrlon)
                if success:
                    drains[name] = Drain(name, cmd, cmdargs)

                    if posobj.type == "rwy" and name.count("/") == 1:
                        aptname, rwyname = name.split('/')
                        drains[name].polys += drawrwy(aptname, [rwyname], posobj.lat, posobj.lon, drawapprwy)


            else:
                try:
                    brg = int(name[:4])
                    drains[name] = Drain(name, cmd, cmdargs)
                    success = True
                except:
                    success = False


        if success:
            if cmd[:6] == "RUNWAY" or cmd == "RWY":
                # Delete old runways
                for polyname in drains[name].polys:
                    stack.stack("DEL "+polyname)
                aptlat, aptlon = drains[name].lat,drains[name].lon
                drains[name].setrunways(cmdargs)
                drains[name].polys += drawrwy(name,cmdargs,aptlat,aptlon,drawapprwy)

            elif cmd=="ORIG":
                success = drains[name].addorig(cmdargs)

            elif cmd=="FLOW":
                sucess = drains[name].setflow(cmdargs[0])
            elif cmd=="TYPES" or cmd=="TYPE":
                drains[name].addactypes(cmdargs)
            elif cmd == "ALT":  # Set fixed alt or interval
                drains[name].setalt(cmdargs)
            elif cmd == "SPD":
                drains[name].setspd(cmdargs)
            elif cmd == "HDG":
                drains[name].sethdg(cmdargs)
            if not success:
                return False, "TRAFGEN SRC ERROR"+cmd+" ".join(cmdargs)

        else:
            return False, "TRAFGEN DRN ERROR " + name + " NOT FOUND"

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

def drawrwy(aptname,cmdargs,aptlat,aptlon,drawfunction):
    rwnames = []

    for rwy in cmdargs:
        if rwy[0] == "R":
            success, rwyposobj = txt2pos(aptname + "/" + rwy, aptlat, aptlon)
        else:
            success, rwyposobj = txt2pos(aptname + "/RW" + rwy, aptlat, aptlon)

        if not success:
            success,rwyposobj = txt2pos(aptname,aptlat,aptlon)

        if success:
            rwydigits = rwy.lstrip("RWY").lstrip("RW")

            # Look up runwayhdg
            try:
                rwyhdg = navdb.rwythresholds[aptname][rwydigits][2]
            except:
                try:
                    rwyhdg = navdb.rwythresholds[aptname][rwydigits.lstrip("0")][2]
                except:
                    rwyhdg = 10.*int(rwydigits.rstrip("LCR").lstrip("0"))

            rwnames.append(drawfunction(aptname, rwy, rwyposobj.lat, rwyposobj.lon, rwyhdg))
        else:
            # Use airport lat,lon en given heading
            stack.stack("ECHO TARFGEN RUNWAY "+aptname+"/"+rwy+" NOT FOUND")

    return rwnames


def drawapprwy(apt,rwy,rwylat,rwylon,rwyhdg):
    # Draw approach ILS arrow
    Lapp  =  7. # [nm] length of approach path drawn
    phi   =  5. # [deg] angle of half the arrow

    # Calculate arrow (T = Threshold runway):
    #                               /------------- L   (left)
    #                 /-----------/              /
    #   T -------------------------------------- A    (approach)
    #                 \-----------\              \
    #                              \--------------R    (right)
    #

    applat,applon     = kwikpos(rwylat,rwylon,(rwyhdg+180)%360.,Lapp)
    rightlat,rightlon = kwikpos(rwylat,rwylon,(rwyhdg+180-phi)%360,Lapp*1.1)
    leftlat,leftlon   = kwikpos(rwylat,rwylon,(rwyhdg+180+phi)%360,Lapp*1.1)

    # Make arguments for POLYLINE command
    T = str(rwylat)+","+str(rwylon)
    A = str(applat)+","+str(applon)
    L = str(leftlat)+","+str(leftlon)
    R = str(rightlat)+","+str(rightlon)

    stack.stack("POLYLINE "+apt+rwy+"-A,"+",".join([T,A,L,T,R,A]))

    return apt+rwy+"-A"

def drawdeprwy(apt,rwy,rwylat,rwylon,rwyhdg):
    # Draw approach ILS arrow
    Ldep  =  5. # [nm] length of approach path drawn
    phi   =  3. # [deg] angle of half the arrow

    # Calculate arrow (T = Threshold runway):
    #      L   (left)
    #     /
    #   D ------------------------------------- T    (approach)
    #     \
    #      R    (right)
    #

    deplat,deplon     = kwikpos(rwylat,rwylon,rwyhdg%360.,Ldep*1.1)
    rightlat,rightlon = kwikpos(rwylat,rwylon,(rwyhdg+phi)%360,Ldep)
    leftlat,leftlon   = kwikpos(rwylat,rwylon,(rwyhdg-phi)%360,Ldep)

    # Make arguments for POLYLINE command
    T = str(rwylat)+","+str(rwylon)
    D = str(deplat)+","+str(deplon)
    L = str(leftlat)+","+str(leftlon)
    R = str(rightlat)+","+str(rightlon)

    stack.stack("POLYLINE " + apt + rwy + "-D," + ",".join([R,D,L,D,T]))
   

    return apt + rwy + "-D"
