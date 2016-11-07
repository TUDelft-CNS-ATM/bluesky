"""
Commandstack module definition : command stack & processing module

Methods:
    Commandstack()          :  constructor
    stack(cmdline)          : add a command to the command stack
    openfile(scenname)      : start playing a scenario file scenname.SCN
                              from scenario folder
    savefile(scenname,traf) : save current traffic situation as
                              scenario file scenname.SCN
    checkfile(t)            : check whether commands need to be
                              processed from scenario file

    process(sim, traf, scr) : central command processing method

Created by  : Jacco M. Hoekstra (TU Delft)
"""
from math import *
import numpy as np
from random import seed
import os
import os.path
import subprocess

from ..tools import geo, areafilter
from ..tools.aero import kts, ft, fpm, tas2cas, density
from ..tools.misc import txt2alt, cmdsplit
from ..tools.position import txt2pos, islat
from .. import settings

# Temporary fix for synthetic
import synthetic as syn

# Global variables
cmddict   = dict()
cmdsynon  = dict()
cmdstack  = []

scenname  = ""
scenfile  = ""
scentime  = []
scencmd   = []

reflat    = -999.  # Reference latitude for searching in nav db in case of duplicate names
reflon    = -999.  # Reference longitude for searching in nav db in case of duplicate names


def init(sim, traf, scr):
    """ Initialization of the default stack commands. This function is called
        at the initialization of the main simulation object."""

    # Command dictionary with command as key, gives a list with:
    #         - helptext
    #         - arglist to specify
    #         - function to call
    #         - description of goal of command
    #
    # Regarding the arglist:
    #    - Separate aruments with a comma ","
    #    - Enclose optional arguments with "[" and "]"
    #    - Separate different argument type variants in one argument with "/"
    #    - Repeat last one using "..." ,    (see e.g. WIND or POLY)
    #
    # Below this dictionary also a dictionary of synonyms is given (equivalent commands)
    #
    #--------------------------------------------------------------------
    commands = {
        "ADDNODES": [
            "ADDNODES number",
            "int",
            sim.addNodes,
            "Add a simulation instance/node"
        ],
        "ADDWPT": [
            "ADDWPT acid, (wpname/lat,lon),[alt],[spd],[afterwp]",
            "acid,wpt,[alt,spd,txt]",
            #
            # lambda *arg: short-hand for using function output as argument, equivalent with:
            #
            # def fun(idx, args):
            #     return traf.ap.route[idx].addwptStack(traf, idx, *args)
            # fun(idx,*args)
            #
            lambda idx, *args: traf.ap.route[idx].addwptStack(traf, idx, *args),
            "Add a waypoint to route of aircraft (FMS)"
        ],
        "ALT": [
            "ALT acid, alt, [vspd]",
            "acid,alt,[vspd]",
            traf.ap.selalt,
            "Altitude command (autopilot)"
        ],
        "AREA": [
            "AREA Shapename/OFF or AREA lat,lon,lat,lon,[top,bottom]",
            "[float/txt,float,float,float,alt,alt]",
            lambda *args: traf.area.setArea(scr, args),
            "Define experiment area (area of interest)"
        ],
        "ASAS": [
            "ASAS ON/OFF",
            "[onoff]",
            traf.asas.toggle,
            "Airborne Separation Assurance System switch"
        ],
        "BATCH": [
            "BATCH filename",
            "string",
            sim.batch,
            "Start a scenario file as batch simulation"
        ],
        "BENCHMARK": [
            "BENCHMARK [scenfile,time]",
            "[txt,time]",
            sim.benchmark,
            "Run benchmark"
        ],
        "BOX": [
            "BOX name,lat,lon,lat,lon,[top,bottom]",
            "txt,latlon,latlon,[alt,alt]",
            lambda name, *coords: areafilter.defineArea(scr, name, 'BOX', coords),
            "Define a box-shaped area"
        ],
        "CALC": [
            "CALC expression",
            "string",
            lambda expr: scr.echo("Ans = " + str(eval(expr))),
            "Simple in-line math calculator, evaluates expression"
        ],
        "CDMETHOD": [
            "CDMETHOD [method]",
            "[txt]",
            traf.asas.SetCDmethod,
            "Set conflict detection method"
        ],
        "CIRCLE": [
            "CIRCLE name,lat,lon,radius,[top,bottom]",
            "txt,latlon,float,[alt,alt]",
            lambda name, *coords: areafilter.defineArea(scr, name, 'CIRCLE', coords),
            "Define a circle-shaped area"
        ],
        "CRE": [
            "CRE acid,type,lat,lon,hdg,alt,spd",
            "txt,txt,latlon,hdg,alt,spd",
            traf.create,
            "Create an aircraft"
        ],
        "DATAFEED":  [
            "DATAFEED [ON/OFF]",
            "[onoff]",
            sim.datafeed
        ],
        "DEL": [
            "DEL acid/WIND/shape",
            "txt",
            lambda a:   traf.delete(a)    if traf.id.count(a) > 0 \
                   else traf.wind.clear() if a=="WIND" \
                   else areafilter.deleteArea(scr, a),
            "Delete command (aircraft, wind, area)"
        ],
        "DELWPT": [
            "DELWPT acid,wpname",
            "acid,txt",
            lambda idx, wpname: traf.ap.route[idx].delwpt(wpname),
            "Delete a waypoint from a route (FMS)"
        ],
        "DEST": [
            "DEST acid, latlon/airport",
            "acid,wpt/latlon",
            lambda idx, *args: traf.ap.setdestorig("DEST", idx, *args),
            "Set destination of aircraft, aircraft wil fly to this airport"
        ],
        "DIRECT": [
            "DIRECT acid wpname",
            "acid,txt",
            lambda idx, wpname: traf.ap.route[idx].direct(traf, idx, wpname),
            "Go direct to specified waypoint in route (FMS)"
        ],
        "DIST": [
            "DIST lat0, lon0, lat1, lon1",
            "latlon,latlon",
            lambda *args: scr.echo("QDR = %.2f deg, Dist = %.3f nm" % geo.qdrdist(*args)),
            "Distance and direction calculation between two positions"
        ],
        "DT": [
            "DT dt",
            "float",
            sim.setDt,
            "Set simulation time step"
        ],
        "DTLOOK": [
            "DTLOOK [time]",
            "[float]",
            traf.asas.SetDtLook,
            "Set lookahead time in seconds for conflict detection"
        ],
        "DTMULT": [
            "DTMULT multiplier",
            "float",
            sim.setDtMultiplier,
            "Sel multiplication factor for fast-time simulation"
        ],
        "DTNOLOOK": [
            "DTNOLOOK [time]",
            "[float]",
            traf.asas.SetDtNoLook,
            "Set interval for conflict detection"
        ],
        "DUMPRTE": [
            "DUMPRTE acid",
            "acid",
            lambda idx: traf.ap.route[idx].dumpRoute(traf, idx),
            "Write route to output/routelog.txt"
        ],
        "ECHO": [
            "ECHO txt",
            "string",
            scr.echo,
            "Show a text in command window for user to read"
        ],
        "ENG": [
            "ENG acid,[engine_id]",
            "acid,[txt]",
            traf.perf.engchange,
            "Specify a different engine type"
        ],
        "FF": [
            "FF [tend]",
            "[time]",
            sim.fastforward,
            "Fast forward the simulation"
        ],
        "FIXDT": [
            "FIXDT ON/OFF [tend]",
            "onoff,[time]",
            sim.setFixdt,
            "Fix the time step"
        ],
        "GETWIND": [
            "GETWIND lat,lon[,alt]",
            "latlon,[alt]",
            traf.wind.get,
            "Get wind at a specified position (and optionally at altitude)"
        ],
        "HDG": [
            "HDG acid,hdg (deg,True)",
            "acid,float",
            traf.ap.selhdg,
            "Heading command (autopilot)"
        ],
        "HELP": [
            "HELP [command]/pdf/ >filename",
            "[txt]",
            lambda *args: scr.echo(showhelp(*args)),
            "Show help on a command, show pdf or write list of commands to file"
        ],
        "HOLD": [
            "HOLD",
            "",
            sim.pause,
            "Pause(hold) simulation"
        ],
        "IC": [
            "IC [IC/filename]",
            "[string]",
            lambda *args: ic(scr, sim, *args),
            "Initial condition: (re)start simulation and open scenario file"
        ],
        "INSEDIT": [
            "INSEDIT txt",
            "string",
            scr.cmdline,
            "Insert text op edit line in command window"
        ],
        "LINE": [
            "LINE name,lat,lon,lat,lon",
            "txt,latlon,latlon",
            lambda name, *coords: scr.objappend(1, name, coords),
            "Draw a line on the radar screen"
        ],
        "LISTRTE": [
            "LISTRTE acid, [pagenr]",
            "acid,[int]",
            lambda idx, *args: traf.ap.route[idx].listrte(scr, idx, traf, *args),
            "Show list of route in window per page of 5 waypoints"
        ],
        "LNAV": [
            "LNAV acid,[ON/OFF]",
            "acid,[onoff]",
            traf.ap.setLNAV,
            "LNAV (lateral FMS mode) switch for autopilot"
        ],
        "MCRE": [
            "MCRE n, [type/*, alt/*, spd/*, dest/*]",
            "int,[txt,alt,spd,txt]",
            lambda *args: traf.mcreate(*args, area=scr.getviewlatlon()),
            "Multiple random create of n aircraft in current view"
        ],
        "METRIC": [
            "METRIC OFF/0/1/2, [dt]",
            "onoff/int,[float]",
            lambda *args: sim.metric.toggle(traf, *args),
            "Complexity metrics module"
        ],
        "MOVE": [
            "MOVE acid,lat,lon,[alt,hdg,spd,vspd]",
            "acid,latlon,[alt,hdg,spd,vspd]",
            traf.move,
            "Move an aircraft to a new position"
        ],
        "ND": [
            "ND acid",
            "txt",
            lambda acid: scr.feature("ND", acid),
            "Show navigation display with CDTI"
        ],
        "NOISE": [
            "NOISE [ON/OFF]",
            "[onoff]",
            traf.setNoise,
            "Turbulence/noise switch"
        ],
        "NOM": [
            "NOM acid",
            "acid",
            traf.nom,
            "Set nominal acceleration for this aircraft (perf model)"
        ],
        "NORESO": [
            "NORESO [acid]",
            "[string]",
            traf.asas.SetNoreso,
            "Switch off conflict resolution for this aircraft"
        ],
        "OP": [
            "OP",
            "",
            sim.start,
            "Start/Run simulation or continue after pause"
        ],
        "ORIG": [
            "ORIG acid, latlon/airport",
            "acid,wpt/latlon",
            lambda *args: traf.ap.setdestorig("ORIG", *args),
            "Set origin airport of aircraft"
        ],
        "PAN": [
            "PAN latlon/acid/airport/waypoint/LEFT/RIGHT/ABOVE/DOWN",
            "pandir/latlon",
            scr.pan,
            "Pan screen (move view) to a waypoint, direction or aircraft"
        ],
        "PCALL": [
            "PCALL filename [REL/ABS]",
            "txt,[txt]",
            lambda *args: openfile(*args, mergeWithExisting=True),
            "Call commands in another scenario file"
        ],

        "POLY": [
            "POLY name,lat,lon,lat,lon, ...",
            "txt,latlon,...",
            lambda name, *coords: areafilter.defineArea(scr, name, 'POLY', coords),
            "Define a polygon-shaped area"
        ],
        "POLYALT": [
            "POLY name,top,bottom,lat,lon,lat,lon, ...",
            "txt,alt,alt,latlon,...",
            lambda name, *coords: areafilter.defineArea(scr, name, 'POLYALT', coords),
            "Define a polygon-shaped area in 3D: between two altitudes"
        ],
        "POS": [
            "POS acid",
            "txt",
            lambda acid: scr.showacinfo(acid, traf.acinfo(acid)),
            "Get info on aircraft"
        ],
        "PRIORULES": [
            "PRIORULES [ON/OFF PRIOCODE]",
            "[onoff, txt]",
            traf.asas.SetPrio,
            "Define priority rules (right of way) for conflict resolution"
        ],
        "QUIT": [
            "QUIT",
            "",
            sim.stop,
            "Quit program/Stop simulation"
        ],
        "RESET": [
            "RESET",
            "",
            sim.reset,
            "Reset simulation"
        ],
        "RFACH": [
            "RFACH [factor]",
            "[float]",
            traf.asas.SetResoFacH,
            "Set resolution factor horizontal (to add a margin)"
        ],
        "RFACV": [
            "RFACV [factor]",
            "[float]",
            traf.asas.SetResoFacV,
            "Set resolution factor vertical (to add a margin)"
        ],
        "RESO": [
            "RESO [method]",
            "[txt]",
            traf.asas.SetCRmethod,
            "Set resolution method"
        ],
        "RESOOFF": [
            "RESOOFF [acid]",
            "[string]",
            traf.asas.SetResooff,
            "Switch for conflict resolution module"
        ],
        "RMETHH": [
            "RMETHH [method]",
            "[txt]",
            traf.asas.SetResoHoriz,
            "Set resolution method to be used horizontally"
        ],
        "RMETHV": [
            "RMETHV [method]",
            "[txt]",
            traf.asas.SetResoVert,
            "Set resolution method to be used vertically"
        ],
        "RSZONEDH": [
            "RSZONEDH [height]",
            "[float]",
            traf.asas.SetPZHm,
            "Set half of vertical dimension of resolution zone in ft"
        ],
        "RSZONER": [
            "RSZONER [radius]",
            "[float]",
            traf.asas.SetPZRm,
            "Set horizontal radius of resolution zone in nm"
        ],
        "RUNWAYS": [
            "RUNWAYS ICAO",
            "txt",
            lambda ICAO: traf.navdb.listrwys(ICAO),
            "List available runways on an airport"
        ],
        "SAVEIC": [
            "SAVEIC filename",
            "string",
            lambda fname: saveic(fname, sim, traf),
            "Save current situation as IC"
        ],
        "SCEN": [
            "SCEN scenname",
            "string",
            scenarioinit,
            "Give current situation a scenario name"
        ],
        "SEED": [
            "SEED value",
            "int",
            setSeed,
            "Set seed for all functions using a randomizer (e.g.mcre,noise)"
        ],
        "SPD": [
            "SPD acid,spd (CAS-kts/Mach)",
            "acid,spd",
            traf.ap.selspd,
            "Speed command (autopilot)"
        ],
        "SSD": [
            "SSD acid/ALL/OFF",
            "txt",
            scr.showssd,
            "Show state-space diagram (=conflict prevention display/predictive ASAS)"
        ],
        "SWRAD": [
            "SWRAD GEO/GRID/APT/VOR/WPT/LABEL/ADSBCOVERAGE/TRAIL [dt]/[value]",
            "txt,[float]",
            scr.feature,
            "Switch on/off elements and background of map/radar view"
        ],
        "SYMBOL": [
            "SYMBOL",
            "",
            scr.symbol,
            "Toggle aircraft symbol"
        ],
        "SYN": [
            " SYN: Possible subcommands: HELP, SIMPLE, SIMPLED, DIFG, SUPER," + \
            "MATRIX, FLOOR, TAKEOVER, WALL, ROW, COLUMN, DISP",
            "txt,[...]",
            lambda *args: syn.process(args[0], len(args) - 1, args, sim, traf, scr),
            "Macro for generating synthetic (geometric) traffic scenarios"
        ],
        "TAXI": [
            "TAXI ON/OFF : OFF auto deletes traffic below 1500 ft",
            "onoff",
            traf.area.setTaxi,
            "Switch on/off ground/low altitude mode, prevents auto-delete at 1500 ft"
        ],
        "TIME": [
            "TIME RUN(default) / HH:MM:SS.hh / REAL / UTC ",
            "[txt]",
            sim.setclock,
            "Set simulated clock time"
        ],
        "TRAIL": [
            "TRAIL ON/OFF, [dt] OR TRAIL acid color",
            "acid/bool,[float/txt]",
            traf.trails.setTrails,
            "Toggle aircraft trails on/off"
        ],
        "VNAV": [
            "VNAV acid,[ON/OFF]",
            "acid,[onoff]",
            traf.ap.setVNAV,
            "Switch on/off VNAV mode, the vertical FMS mode (autopilot)"
        ],
        "VS": [
            "VS acid,vspd (ft/min)",
            "acid,vspd",
            traf.ap.selvspd,
            "Vertical speed command (autopilot)"
        ],
        "WIND": [
            "WIND lat,lon,alt/*,dir,spd[,alt,dir,spd,alt,...]",
            "latlon,[alt],float,float,...,...,...",   # last 3 args are repeated
            traf.wind.add,
            "Define a wind vector as part of the 2D or 3D wind field"
        ],
        "ZONEDH": [
            "ZONEDH [height]",
            "[float]",
            traf.asas.SetPZH,
            "Set half of the vertical protected zone dimensions in ft"
        ],
        "ZONER": [
            "ZONER [radius]",
            "[float]",
            traf.asas.SetPZR,
            "Set the radius of the horizontal protected zone dimensions in nm"

        ],
        "ZOOM": [
            "ZOOM IN/OUT or factor",
            "float/txt",
            lambda a: scr.zoom(1.4142135623730951) if a == "IN" else \
                      scr.zoom(0.7071067811865475) if a == "OUT" else \
                      scr.zoom(a, True),
            "Zoom display in/out, you can also use +++ or -----"
        ]
    }

    cmddict.update(commands)

    #--------------------------------------------------------------------
    # Command synonym dictionary
    synonyms = {
        "CONTINUE": "OP",
        "CREATE": "CRE",
        "CLOSE": "QUIT",
        "DELETE": "DEL",
        "DIRECTTO": "DIRECT",
        "DIRTO": "DIRECT",
        "DISP": "SWRAD",
        "END": "QUIT",
        "EXIT": "QUIT",
        "FWD": "FF",
        "HMETH": "RMETHH",
        "HRESOM": "RMETHH",
        "HRESOMETH": "RMETHH",
        "LOAD": "IC",
        "OPEN": "IC",
        "PAUSE": "HOLD",
        "Q": "QUIT",
        "STOP": "QUIT",
        "RUN": "OP",
        "RESOFACH": "RFACH",
        "RESOFACV": "RFACV",
        "SAVE": "SAVEIC",
        "START": "OP",
        "TURN": "HDG",
        "VMETH": "RMETHV",
        "VRESOM": "RMETHV",
        "VRESOMETH": "RMETHV",
        "?": "HELP"
    }
    cmdsynon.update(synonyms)
    #--------------------------------------------------------------------

    # Display Help text on start of program
    stack("ECHO BlueSky Console Window: Enter HELP or ? for info.\n" +
        "Or select IC to Open a scenario file.")

    # Pan to initial location
    stack('PAN ' + settings.start_location)
    stack("ZOOM 0.4")


def get_scenname():
    return scenname


def get_scendata():
    return scentime, scencmd


def set_scendata(newtime, newcmd):
    global scentime, scencmd
    scentime = newtime
    scencmd  = newcmd


def scenarioinit(name):
    global scenname
    scenname = name
    return True, 'Starting scenario ' + name


def append_commands(newcommands):
    """ Append additional functions to the stack command dictionary """
    cmddict.update(newcommands)


def showhelp(cmd=''):
    """ Generate help text for displaying or dump command reference in file
        when command is >filename
    """
    # No command given: show all
    if len(cmd) == 0:
        text = "To get help on a command, enter it without arguments.\n" + \
               "The BlueSky commands are:\n\n"
        text2 = ""
        for key in cmddict:
            text2 += (key + " ")
            if len(text2) >= 60:
                text += (text2 + "\n")
                text2 = ""
        text += (text2 + "\nSee Info subfolder for more info.")
        return text

    elif cmd.upper()=="PDF":
        cwd = os.getcwd()
        os.chdir("info")
        pdfhelp = "BLUESKY-COMMAND-TABLE.pdf"
        if os.path.isfile(pdfhelp):
            try:
                subprocess.Popen(pdfhelp,shell=True)
            except:
                return "Opening "+pdfhelp+" failed."
        else:
            return pdfhelp+"does not exist."
        os.chdir("..")
        return "Pdf window opened"

    # Show help line for command
    elif cmd in cmddict:

        # Check whether description is available, then show it as well
        if len(cmddict)<=3:
            return cmddict[cmd][0]
        else:
            return cmddict[cmd][0]+"\n"+cmddict[cmd][3]

    # Show help line for equivalent command
    elif cmd in cmdsynon:

        # Check whether description is available, then show it as well
        if len(cmddict[cmdsynon[cmd]])<=3:
            return cmddict[cmdsynon[cmd]][0]
        else:
            return cmddict[cmdsynon[cmd]][0]+"\n"+cmddict[cmdsynon[cmd]][3]


    # Write command reference to tab-delimited text file
    elif cmd[0] == ">":

        # Get filename
        if len(cmd)>1:
            fname = "./info/"+cmd[1:]
        else:
            fname = "./info/bluesky-commands.txt"

        # Write command dictionary to tab-dleoimited text file
        try:
            f = open(fname,"w")
        except:
            return "Invalid filename:"+fname

        # Header of first table
        f.write("Command\tDescription\tUsage\tArgument types\tFunction\n")


        table = []  # for alphabetical sort use a table

        # Get info for all commands
        for item in cmddict:

            lst = cmddict[item] # Get list with helpline, argtypes & function

            line = item + "\t"
            if len(lst)>3:
                line = line + lst[3]
            line = line + "\t" + lst[0] + "\t" + str(lst[1]) + "\t"

            # Clean up string with function name and add if not a lambda function
            funct = str(lst[2]).replace("<","").replace(">","")

            # Lambda function give no info, also remove hex address and "method" text
            if funct.count("lambda")==0:

                if funct.count("at")>0:
                    idxat = funct.index(" at ")
                    funct = funct[:idxat]

                funct = funct.replace("bound method","")
                line = line + funct

            table.append(line)

        # Sort & write table
        table.sort()
        for line in table:
            f.write(line+"\n")
        f.write("\n")

        # Add synonyms table
        f.write("\n\n Synonyms (equivalent commands)\n")

        table = []  # for alphabetical sort use table
        for item in cmdsynon:
            if cmdsynon[item] in cmddict and len(cmddict[cmdsynon[item]])>=3 :
                table.append(item + "\t" +cmdsynon[item]+"\t"+cmddict[cmdsynon[item]][3])
            else:
                table.append(item + "\t" +cmdsynon[item]+"\t")


        # Sort & write table
        table.sort()
        for line in table:
            f.write(line+"\n")
        f.write("\n")

        # Close and report where file is to be found
        f.close()
        return "Writing command reference in "+fname

    else:
        return "HELP: Unknown command: " + cmd


def setSeed(value):
    seed(value)
    np.random.seed(value)


def reset():
    global scentime, scencmd, scenname

    scentime = []
    scencmd  = []
    scenname = ''


def stack(cmdline):
    # Stack one or more commands separated by ";"
    cmdline = cmdline.strip()
    if len(cmdline) > 0:
        for line in cmdline.split(';'):
            cmdstack.append(line)


def openfile(fname, absrel='ABS', mergeWithExisting=False):
    global scentime, scencmd

    # Split the incoming filename into a path, a filename and an extension
    path, fname   = os.path.split(os.path.normpath(fname))
    scenname, ext = os.path.splitext(fname)
    if len(path) == 0:
        path = os.path.normpath(settings.scenario_path)
    if len(ext) == 0:
        ext = '.scn'

    # The entire filename, possibly with added path and extension
    scenfile = os.path.join(path, scenname + ext)

    print "Opening ", scenfile

    # If timestamps in file should be interpreted as relative we need to add
    # the current simtime to every timestamp
    t_offset = sim.simt if absrel == 'REL' else 0.0

    if not os.path.exists(scenfile):
        return False, "Error: cannot find file: " + scenfile

    # Split scenario file line in times and commands
    if not mergeWithExisting:
        # When a scenario file is read with PCALL the resulting commands
        # need to be merged with the existing commands. Otherwise the
        # old scenario commands are cleared.
        scentime = []
        scencmd  = []

    with open(scenfile, 'r') as fscen:
        for line in fscen:
            if len(line.strip()) > 12 and line[0] != "#":
                # Try reading timestamp and command
                try:
                    icmdline = line.index('>')
                    tstamp   = line[:icmdline]
                    ttxt     = tstamp.strip().split(':')
                    ihr      = int(ttxt[0]) * 3600.0
                    imin     = int(ttxt[1]) * 60.0
                    xsec     = float(ttxt[2])
                    scentime.append(ihr + imin + xsec + t_offset)
                    scencmd.append(line[icmdline + 1:].strip("\n"))
                except:
                    if not(len(line.strip()) > 0 and line.strip()[0] == "#"):
                        print "except this:", line
                    pass  # nice try, we will just ignore this syntax error

    if mergeWithExisting:
        # If we are merging we need to sort the resulting command list
        scentime, scencmd = [list(x) for x in zip(*sorted(
            zip(scentime, scencmd), key=lambda pair: pair[0]))]

    return True


def ic(scr, sim, filename=''):
    global scenfile, scenname

    # Get the filename of new scenario
    if filename == '':
        filename = scr.show_file_dialog()
    elif filename == "IC":
        filename = scenfile

    # Clean up filename
    try:
        filename = filename.strip()
    except:
        pass

    # Reset sim and open new scenario file
    if len(filename) > 0:
        sim.reset()
        result = openfile(filename)
        if result is True:
            scenfile    = filename
            scenname, _ = os.path.splitext(os.path.basename(filename))
            return True, "Opened " + filename
        else:
            return result


def checkfile(simt):
    # Empty command buffer when it's time
    while len(scencmd) > 0 and simt >= scentime[0]:
        stack(scencmd[0])
        del scencmd[0]
        del scentime[0]

    return


def saveic(fname, sim, traf):
    # Add extension .scn if not already present
    if fname.lower().find(".scn") < 0:
        fname = fname + ".scn"

    # If it is with path don't touch it, else add path
    if fname.find("/") < 0:
        scenfile = "./scenario/" + fname

    try:
        f = open(scenfile, "w")
    except:
        return False, "Error writing to file"

    # Write files
    timtxt = "00:00:00.00>"

    for i in range(traf.ntraf):
        # CRE acid,type,lat,lon,hdg,alt,spd
        cmdline = "CRE " + traf.id[i] + "," + traf.type[i] + "," + \
                  repr(traf.lat[i]) + "," + repr(traf.lon[i]) + "," + \
                  repr(traf.trk[i]) + "," + repr(traf.alt[i] / ft) + "," + \
                  repr(tas2cas(traf.tas[i], traf.alt[i]) / kts)

        f.write(timtxt + cmdline +"\n")

        # VS acid,vs
        if abs(traf.vs[i]) > 0.05:  # 10 fpm dead band
            if abs(traf.ap.vs[i]) > 0.05:
                vs_ = traf.ap.vs[i] / fpm
            else:
                vs_ = traf.vs[i] / fpm

            cmdline = "VS " + traf.id[i] + "," + repr(vs_)
            f.write(timtxt + cmdline + "\n")

        # Autopilot commands
        # Altitude
        if abs(traf.alt[i] - traf.ap.alt[i]) > 10.:
            cmdline = "ALT " + traf.id[i] + "," + repr(traf.ap.alt[i] / ft)
            f.write(timtxt + cmdline + "\n")

        # Heading as well when heading select
        delhdg = (traf.hdg[i] - traf.ap.trk[i] + 180.) % 360. - 180.
        if abs(delhdg) > 0.5:
            cmdline = "HDG " + traf.id[i] + "," + repr(traf.ap.trk[i])
            f.write(timtxt + cmdline + "\n")

        # Speed select? => Record
        rho = density(traf.alt[i])  # alt in m!
        aptas = sqrt(1.225 / rho) * traf.ap.spd[i]
        delspd = aptas - traf.tas[i]

        if abs(delspd) > 0.4:
            cmdline = "SPD " + traf.id[i] + "," + repr(traf.ap.spd[i] / kts)
            f.write(timtxt + cmdline + "\n")

        # DEST acid,dest-apt
        if traf.ap.dest[i] != "":
            cmdline = "DEST " + traf.id[i] + "," + traf.ap.dest[i]
            f.write(timtxt + cmdline + "\n")

        # ORIG acid,orig-apt
        if traf.ap.orig[i] != "":
            cmdline = "ORIG " + traf.id[i] + "," + traf.ap.orig[i]
            f.write(timtxt + cmdline + "\n")

        # Route with ADDWPT
        route = traf.ap.route[i]
        for iwp in range(route.nwp):
            # dets and orig al already done, skip them here
            if iwp==0 and route.wpname[iwp]==traf.ap.orig[i]:
                continue

            if iwp==route.nwp-1 and route.wpname[iwp]==traf.ap.dest[i]:
                continue

            #add other waypoints
            cmdline = "ADDWPT "+traf.id[i]+" "
            wpname = route.wpname[iwp]
            if wpname[:len(traf.id[i])]==traf.id[i]:
                wpname = repr(route.lat[iwp])+","+repr(route.lon[iwp])
            cmdline = cmdline + wpname+","

            if route.wpalt[iwp]>=0.:
                cmdline = cmdline +repr(route.wpalt[iwp])+","
            else:
                cmdline = cmdline+","

            if route.wpspd[iwp]>=0.:
                cmdline = cmdline +repr(route.wpspd[iwp])+","

            f.write(timtxt + cmdline + "\n")

    # Saveic: should close
    f.close()
    return True


def process(sim, traf, scr):
    """process and empty command stack"""
    global cmdstack

    # Process stack of commands
    for line in cmdstack:
        #debug       print "stack is processing:",line
        # Empty line: next command
        line = line.strip()
        if len(line) == 0:
            continue

        # Split command line into command and arguments, pass traf ids to check for
        # switched acid and command
        cmd, args = cmdsplit(line.upper(), traf.id)
        numargs   = len(args)
        # Check if this is a POS command with only an aircraft id
        if numargs == 0 and traf.id.count(cmd) > 0:
            args    = [cmd]
            cmd     = 'POS'
            numargs = 1

        # Assume syntax is ok (default)
        synerr = False

        #**********************************************************************
        #=====================  Start of command branches =====================
        #**********************************************************************

        #----------------------------------------------------------------------
        # First check command synonymes list, then in dictionary
        #----------------------------------------------------------------------
        orgcmd = cmd # save for string cutting out of line and use of synonyms
        if cmd in cmdsynon.keys():
            cmd    = cmdsynon[cmd]


        if cmd in cmddict.keys():
            helptext, argtypelist, function = cmddict[cmd][:3]
            argvsopt = argtypelist.split('[')
            argtypes = argvsopt[0].strip(',').split(",")
            if argtypes == ['']:
                argtypes = []

            # Check if at least the number of mandatory arguments is given.
            if numargs < len(argtypes):
                scr.echo("Syntax error: Too few arguments")
                scr.echo(line)
                scr.echo(helptext)
                continue

            # Add optional argument types if they are given
            if len(argvsopt) == 2:
                argtypes = argtypes + argvsopt[1].strip(']').split(',')

            # Process arg list
            optargs = {}
            # Special case: single text string argument: case sensitive,
            # possibly with spaces/newlines pass the original
            if argtypes == ['string']:
                arglist = [line[len(orgcmd) + 1:]]
            else:
                arglist = []
                curtype = curarg = 0
                while curtype < len(argtypes) and curarg < len(args) and not synerr:
                    if argtypes[curtype][:3] == '...':
                        repeatsize = len(argtypes) - curtype
                        curtype = curtype - repeatsize
                    argtype    = argtypes[curtype].strip().split('/')

                    # Go over all argtypes separated by"/" in this place in the command line
                    for i in range(len(argtype)):
                        argtypei = argtype[i]
                        parsed_arg, opt_arg, argstep = argparse(argtypei, curarg, args, traf, scr)

                        if parsed_arg[0] is None:
                            # not yet last type possible here?
                            if i < len(argtype) - 1:
                                # We have alternative argument formats that we can try
                                continue
                            elif argtypei in optargs:
                                # Missing arguments, so maybe not filled in so enter optargs?
                                arglist += optargs[argtypei]
                            else:
                                synerr = True
                                scr.echo("Syntax error in processing arguments")
                                scr.echo(line)
                                scr.echo(helptext)
                                print "Error in processing arguments:"
                                print line
                        else:
                            arglist += parsed_arg

                        optargs.update(opt_arg)
                        curarg  += argstep
                        break

                    curtype += 1

            # Call function return flag,text
            # flag: indicates sucess
            # text: optional error message
            if not synerr:
                # print cmd, arglist
                results = function(*arglist)  # * = unpack list to call arguments

                if type(results) == bool:  # Only flag is returned
                    synerr = not results
                    if synerr:
                        if numargs <= 0 or curarg < len(args) and args[curarg] == "?":
                            scr.echo(helptext)
                        else:
                            scr.echo("Syntax error: " + helptext)
                        synerr =  False  # Prevent further nagging

                elif type(results) == list or type(results) == tuple:
                    # Maybe there is also an error message returned?
                    if len(results) >= 1:
                        synerr = not results[0]

                    if len(results) >= 2:
                        scr.echo(cmd + ":" + results[1])
                        synerr = False

            else:  # synerr:
                scr.echo("Syntax error: " + helptext)

        #----------------------------------------------------------------------
        # ZOOM command (or use ++++  or --  to zoom in or out)
        #----------------------------------------------------------------------
        elif cmd[0] in ["+", "=", "-"]:
            nplus = cmd.count("+") + cmd.count("=")  # = equals + (same key)
            nmin  = cmd.count("-")
            scr.zoom(sqrt(2) ** (nplus - nmin), absolute=False)

        #-------------------------------------------------------------------
        # Reference to other command files
        # Check external references
        #-------------------------------------------------------------------
#        elif cmd[:4] in extracmdrefs:
#            extracmdrefs[cmd[:4]].process(cmd[4:], numargs, [cmd] + args, sim, traf, scr, self)

        #-------------------------------------------------------------------
        # Command not found
        #-------------------------------------------------------------------
        else:
            if numargs == 0:
                scr.echo("Unknown command or aircraft: " + cmd)
            else:
                scr.echo("Unknown command: " + cmd)

        #**********************************************************************
        #======================  End of command branches ======================
        #**********************************************************************

    # End of for-loop of cmdstack
    cmdstack = []
    return


def argparse(argtype, argidx, args, traf, scr):
    global reflat, reflon
    """ Parse one or more arguments.

        Returns:
        - A list with the parse results
        - The number of arguments parsed
        - A dict with additional optional parsed arguments.
        As different types can be tried, return none if syntax not ok"""

    if args[argidx] == "" or args[argidx] == "*":  # Empty arg or wildcard => parse None
        return [None], {}, 1

    elif argtype == "acid":  # aircraft id => parse index
        idx = traf.id2idx(args[argidx])
        if idx < 0:
            scr.echo(args[idx] + " not found")
            return [-1], {}, 1
        else:
            reflat, reflon = traf.lat[idx], traf.lon[idx]  # Update ref position for navdb lookup
            return [idx], {}, 1

    elif argtype == "txt":  # simple text
        return [args[argidx]], {}, 1

    elif argtype == "float":  # float number
        return [float(args[argidx])], {}, 1

    elif argtype == "int":   # integer
        return [int(args[argidx])], {}, 1

    elif argtype == "onoff" or argtype == "bool":
        sw = (args[argidx] == "ON" or
              args[argidx] == "1" or args[argidx] == "TRUE")
        return [sw], {}, 1

    elif argtype == "wpt" or argtype == "latlon":

        # wpt: Make 1 or 2 argument(s) into 1 position text to be used as waypoint
        # latlon: return lat,lon to be used as a position only

        # Examples valid position texts:
        # lat/lon : "N52.12,E004.23","N52'14'12',E004'23'10"
        # navaid/fix: "SPY","OA","SUGOL"
        # airport:   "EHAM"
        # runway:    "EHAM/RW06" "LFPG/RWY23"

        # Set default lat,lon to screen
        if reflat < 180.:  # No reference avaiable yet: use screen center
            reflat, reflon = scr.ctrlat, scr.ctrlon

        optargs = {}

        # If last argument, no lat,lom or airport,runway so simply return this argument
        if len(args) - 1 == argidx:
            # translate a/c id into a valid position text with a lat,lon
            if traf.id2idx(args[argidx]) >= 0:
                idx  = traf.id2idx(args[argidx])
                name = str(traf.lat[idx]) + "," + str(traf.lon[idx])
            else:
                name = args[argidx]
            nusedargs = 1  # we used one argument

        # Check occasionally also next arg
        else:
            # lat,lon ? Combine into one string with a comma
            if islat(args[argidx]):
                name = args[argidx] + "," + args[argidx + 1]
                nusedargs = 2   # we used two arguments

            # apt,runway ? Combine into one string with a slash as separator
            elif args[argidx + 1][:2].upper() == "RW" and traf.navdb.apid.count(args[argidx]) > 0:
                name = args[argidx] + "/" + args[argidx + 1]
                nusedargs = 2   # we used two arguments

            # aircraft id? convert to lat/lon string
            elif traf.id2idx(argidx) >= 0:
                idx = traf.id2idx(args[argidx])
                name = str(traf.lat[idx]) + "," + str(traf.lon[idx])
                nusedargs = 1

            # In other cases parse string as position
            else:
                name = args[argidx]
                nusedargs = 1  # we used one argument

        # Return something different for the two argtypes:

        # for wpt argument type, simply return positiontext, no need it look up nw
        if argtype == "wpt":
            return [name], optargs, nusedargs

        # for lat/lon argument type we also need to it up:
        elif argtype == "latlon":
            success, posobj = txt2pos(name, traf, traf.navdb, reflat, reflon)

            if success:
                # for runway type, get heading as default optional argument for command line
                if posobj.type == "rwy":
                    rwyname = args[argidx + 1].strip("RW").strip("Y").strip().upper()  # remove RW or RWY and spaces
                    optargs = {"hdg": [traf.navdb.rwythresholds[args[argidx]][rwyname][2]]}

                reflat, reflon = posobj.lat, posobj.lon

                return [posobj.lat , posobj.lon], optargs, nusedargs
            else:
                scr.echo(posobj)  # contains error message
                return [None], {}, 1

    elif argtype == "pandir":  # Pan direction

        if args[argidx].upper().strip() in ["LEFT", "RIGHT", "UP", "ABOVE", "RIGHT", "DOWN"]:
            return [args[argidx].upper()], {}, 1  # pass on string to pan function
        else:
            return [None], {}, 1

    elif argtype == "spd":  # CAS[kts] Mach
        spd = float(args[argidx].upper().replace("M", ".").replace("..", "."))
        if not 0.1 < spd < 1.0:
            spd *= kts
        return [spd], {}, 1  # speed CAS[m/s] or Mach (float)

    elif argtype == "vspd":
        try:
            return [fpm * float(args[argidx])], {}, 1
        except:
            return [None],{}, 0


    elif argtype == "alt":  # alt: FL250 or 25000 [ft]
        return [ft * txt2alt(args[argidx])], {}, 1  # alt in m

    elif argtype == "hdg":
        # TODO: for now no difference between magnetic/true heading
        hdg = float(args[argidx].upper().replace('T', '').replace('M', ''))
        return [hdg], {}, 1

    elif argtype == "time":
        ttxt = args[argidx].strip().split(':')
        if len(ttxt) >= 3:
            ihr  = int(ttxt[0]) * 3600.0
            imin = int(ttxt[1]) * 60.0
            xsec = float(ttxt[2])
            return [ihr + imin + xsec], {}, 1
        else:
            try:
                return [float(args[argidx])], {}, 1
            except:
                return [None],{}, 0

    return [None],{}, 0
