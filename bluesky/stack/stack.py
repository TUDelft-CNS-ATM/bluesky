"""
Commandstack module definition : command stack & processing module

Methods:
    Commandstack()          :  constructor
    stack(cmdline)          : add a command to the command stack
    openfile(scenname)      : start playing a scenario file scenname.SCN
                              from scenario folder
    savefile(scenname)      : save current traffic situation as
                              scenario file scenname.SCN
    checkfile(t)            : check whether commands need to be
                              processed from scenario file

    process()               : central command processing method

Created by  : Jacco M. Hoekstra (TU Delft)
"""
from math import *
from random import seed
import os
import os.path
import subprocess
import numpy as np
import bluesky as bs
from bluesky.tools import geo, areafilter, plugin
from bluesky.tools.aero import kts, ft, fpm, tas2cas, density
from bluesky.tools.misc import txt2alt, cmdsplit
from bluesky.tools.calculator import calculator
from bluesky.tools.position import txt2pos, islat
from bluesky import settings

# Temporary fix for synthetic
from . import synthetic as syn
# Register settings defaults
settings.set_variable_defaults(start_location='EHAM', scenario_path='scenario')

# Global variables
cmddict   = dict()  # Defined in stack.init

#
# Command synonym dictionary definea equivalent commands globally in stack
#
# Actual command definitions: see dictionary in def init(...) below
#
cmdsynon  = {"ADDAIRWAY": "ADDAWY",
             "AWY": "POS",
             "AIRPORT": "POS",
             "AIRWAYS": "AIRWAY",
             "CALL": "PCALL",
             "CONTINUE": "OP",
             "CREATE": "CRE",
             "CLOSE": "QUIT",
             "DELETE": "DEL",
             "DELWP": "DELWPT",
             "DELROUTE": "DELRTE",
             "DIRECTTO": "DIRECT",
             "DIRTO": "DIRECT",
             "DISP": "SWRAD",
             "END": "QUIT",
             "EXIT": "QUIT",
             "FWD": "FF",
             "HEADING": "HDG",
             "HMETH": "RMETHH",
             "HRESOM": "RMETHH",
             "HRESOMETH": "RMETHH",
             "LOAD": "IC",
             "OPEN": "IC",
             "PAUSE": "HOLD",
             "Q": "QUIT",
             "STOP": "QUIT",
             "RUN": "OP",
             "RUNWAYS": "POS",
             "RESOFACH": "RFACH",
             "RESOFACV": "RFACV",
             "SAVE": "SAVEIC",
             "SPEED": "SPD",
             "START": "OP",
             "TRAILS": "TRAIL",
             "TURN": "HDG",
             "VMETH": "RMETHV",
             "VRESOM": "RMETHV",
             "VRESOMETH": "RMETHV",
             "?": "HELP"
            }


cmdstack  = []

scenname  = ""
scenfile  = ""
scentime  = []
scencmd   = []


def init():
    """ Initialization of the default stack commands. This function is called
        at the initialization of the main simulation object."""

    # Command dictionary with command as key, gives a list with:
    #
    #         command: [ helptext ,
    #                    arglist ,
    #                    function to call,
    #                    description in one line ]
    #
    # Regarding the arglist:
    #    - Separate aruments with a comma ","
    #    - Enclose optional arguments with "[" and "]"
    #    - Separate different argument type variants in one argument with "/"
    #    - Repeat last one using "..." ,    (see e.g. WIND or POLY)
    #
    # Argtypes = syntax parsing (see below in this module for parsing):
    #
    #   acid      = aircraft id (text => index)
    #   alt       = altitude (FL250, 25000  ft+. meters)
    #   spd       = CAS or Mach (when <1)   => m/s
    #   hdg       = heading in degrees
    #
    #   float     = plain float
    #   int       = integer
    #   txt       = string
    #   on/off    = text => boolean
    #
    #   latlon    = converts acid, wpt, airport etc => lat,lon (deg) so 2 args!
    #   wpt       = converts postext or lat,lon into a text string to be used as named waypoint
    #   wpinroute = text string with name of waypoint in route
    #   pandir    = text with LEFT, RIGHT, UP/ABOVE or DOWN
    #
    # Below this dictionary also a dictionary of synonym commandss is given (equivalent commands)
    #
    #--------------------------------------------------------------------
    commands = {
        "ADDNODES": [
            "ADDNODES number",
            "int",
            bs.sim.addNodes,
            "Add a simulation instance/node"
        ],
        "ADDWPT": [
            "ADDWPT acid, (wpname/lat,lon/FLYBY/FLYOVER/ TAKEOFF,APT/RWY),[alt,spd,afterwp]",
            "acid,wpt,[alt/txt,spd,wpinroute]",
            #
            # lambda *arg: short-hand for using function output as argument, equivalent with:
            #
            # def fun(idx, args):
            #     return bs.traf.ap.route[idx].addwptStack(idx, *args)
            # fun(idx,*args)
            #
            lambda idx, *args: bs.traf.ap.route[idx].addwptStack(idx, *args),
            "Add a waypoint to route of aircraft (FMS)"
        ],
        "AFTER": [
            "acid AFTER afterwp ADDWPT (wpname/lat,lon),[alt,spd]",
            "acid,wpinroute,txt,wpt,[alt,spd]",
            lambda idx, *args: bs.traf.ap.route[idx].afteraddwptStack(idx, *args),
            "After waypoint, add a waypoint to route of aircraft (FMS)"
        ],
        "AIRWAY": [
            "AIRWAY wp/airway",
            "txt",
            bs.traf.airwaycmd,
            "Get info on airway or connections of a waypoint"
        ],
        "ALT": [
            "ALT acid, alt, [vspd]",
            "acid,alt,[vspd]",
            bs.traf.ap.selaltcmd,
            "Altitude command (autopilot)"
        ],
        "ASAS": [
            "ASAS ON/OFF",
            "[onoff]",
            bs.traf.asas.toggle,
            "Airborne Separation Assurance System switch"
        ],
        "AT": [
            "acid AT wpname [DEL] SPD/ALT [spd/alt]",
            "acid,wpinroute,[txt,txt]",
            lambda idx, *args: bs.traf.ap.route[idx].atwptStack(idx, *args),
            "Edit, delete or show spd/alt constraints at a waypoint in the route"
        ],
        "BATCH": [
            "BATCH filename",
            "string",
            bs.sim.batch,
            "Start a scenario file as batch simulation"
        ],
        "BENCHMARK": [
            "BENCHMARK [scenfile,time]",
            "[txt,time]",
            bs.sim.benchmark,
            "Run benchmark"
        ],
        "BOX": [
            "BOX name,lat,lon,lat,lon,[top,bottom]",
            "txt,latlon,latlon,[alt,alt]",
            lambda name, *coords: areafilter.defineArea(name, 'BOX', coords[:4], *coords[4:]),
            "Define a box-shaped area"
        ],
        "CALC": [
            "CALC expression",
            "string",
            calculator,
            "Simple in-line math calculator, evaluates expression"
        ],
        "CDMETHOD": [
            "CDMETHOD [method]",
            "[txt]",
            bs.traf.asas.SetCDmethod,
            "Set conflict detection method"
        ],
        "CIRCLE": [
            "CIRCLE name,lat,lon,radius,[top,bottom]",
            "txt,latlon,float,[alt,alt]",
            lambda name, *coords: areafilter.defineArea(name, 'CIRCLE', coords[:3], *coords[3:]),
            "Define a circle-shaped area"
        ],
        "CRE": [
            "CRE acid,type,lat,lon,hdg,alt,spd",
            "txt,txt,latlon,hdg,alt,spd",
            bs.traf.create,
            "Create an aircraft"
        ],
        "CRECONFS": [
            "CRECONFS id, type, targetid, dpsi, cpa, tlos_hor, dH, tlos_ver, spd",
            "txt,txt,acid,hdg,float,time,[alt,time,spd]",
            bs.traf.creconfs,
            "Create an aircraft that is in conflict with 'targetid'"
        ],
        "DEFWPT": [
            "DEFWPT wpname,lat,lon,[FIX/VOR/DME/NDB]",
            "txt,latlon,[txt,txt,txt]",
            bs.navdb.defwpt,
            "Define a waypoint only for this scenario/run"
        ],
        "DEL": [
            "DEL acid/WIND/shape",
            "txt",
            lambda a:   bs.traf.delete(a)    if bs.traf.id.count(a) > 0 \
                   else bs.traf.wind.clear() if a == "WIND" \
                   else areafilter.deleteArea(a),
            "Delete command (aircraft, wind, area)"
        ],
        "DELAY": [
            "DELAY time offset, COMMAND+ARGS",
            "time,txt,...",
            lambda time,*args: sched_cmd(time, args, relative=True),
            "Add a delayed command to stack"
        ],
        "DELRTE": [
            "DELRTE acid",
            "acid",
            lambda idx: bs.traf.ap.route[idx].delrte(),
            "Delete for this a/c the complete route/dest/orig (FMS)"
        ],
        "DELWPT": [
            "DELWPT acid,wpname",
            "acid,wpinroute",
            lambda idx, wpname: bs.traf.ap.route[idx].delwpt(wpname),
            "Delete a waypoint from a route (FMS)"
        ],
        "DEST": [
            "DEST acid, latlon/airport",
            "acid,wpt/latlon",
            lambda idx, *args: bs.traf.ap.setdestorig("DEST", idx, *args),
            "Set destination of aircraft, aircraft wil fly to this airport"
        ],
        "DIRECT": [
            "DIRECT acid wpname",
            "acid,txt",
            lambda idx, wpname: bs.traf.ap.route[idx].direct(idx, wpname),
            "Go direct to specified waypoint in route (FMS)"
        ],
        "DIST": [
            "DIST lat0, lon0, lat1, lon1",
            "latlon,latlon",
            distcalc,
            "Distance and direction calculation between two positions"
        ],
        "DOC": [
            "DOC [command]",
            "[txt]",
            bs.scr.show_cmd_doc,
            "Show extended help window for given command, or the main documentation page if no command is given."
        ],
        "DT": [
            "DT dt",
            "float",
            bs.sim.setDt,
            "Set simulation time step"
        ],
        "DTLOOK": [
            "DTLOOK [time]",
            "[float]",
            bs.traf.asas.SetDtLook,
            "Set lookahead time in seconds for conflict detection"
        ],
        "DTMULT": [
            "DTMULT multiplier",
            "float",
            bs.sim.setDtMultiplier,
            "Sel multiplication factor for fast-time simulation"
        ],
        "DTNOLOOK": [
            "DTNOLOOK [time]",
            "[float]",
            bs.traf.asas.SetDtNoLook,
            "Set interval for conflict detection"
        ],
        "DUMPRTE": [
            "DUMPRTE acid",
            "acid",
            lambda idx: bs.traf.ap.route[idx].dumpRoute(idx),
            "Write route to output/routelog.txt"
        ],
        "ECHO": [
            "ECHO txt",
            "string",
            bs.scr.echo,
            "Show a text in command window for user to read"
        ],
        "ENG": [
            "ENG acid,[engine_id]",
            "acid,[txt]",
            bs.traf.perf.engchange,
            "Specify a different engine type"
        ],
        "FF": [
            "FF [timeinsec]",
            "[time]",
            bs.sim.fastforward,
            "Fast forward the simulation"
        ],
        "FILTERALT":
        [
            "FILTERALT ON/OFF,[bottom,top]",
            "bool,[alt,alt]",
            bs.scr.filteralt,
            "Display aircraft on only a selected range of altitudes"
        ],
        "FIXDT": [
            "FIXDT ON/OFF [tend]",
            "onoff,[time]",
            bs.sim.setFixdt,
            "Fix the time step"
        ],
        "GETWIND": [
            "GETWIND lat,lon,[alt]",
            "latlon,[alt]",
            bs.traf.wind.get,
            "Get wind at a specified position (and optionally at altitude)"
        ],
        "HDG": [
            "HDG acid,hdg (deg,True)",
            "acid,float",
            bs.traf.ap.selhdgcmd,
            "Heading command (autopilot)"
        ],
        "HELP": [
            "HELP [command]/pdf/ >filename",
            "[txt]",
            lambda *args: bs.scr.echo(showhelp(*args)),
            "Show help on a command, show pdf or write list of commands to file"
        ],
        "HOLD": [
            "HOLD",
            "",
            bs.sim.pause,
            "Pause(hold) simulation"
        ],
        "IC": [
            "IC [IC/filename]",
            "[string]",
            ic,
            "Initial condition: (re)start simulation and open scenario file"
        ],
        "INSEDIT": [
            "INSEDIT txt",
            "string",
            bs.scr.cmdline,
            "Insert text op edit line in command window"
        ],
        "LINE": [
            "LINE name,lat,lon,lat,lon",
            "txt,latlon,latlon",
            lambda name, *coords: bs.scr.objappend("LINE", name, coords),
            "Draw a line on the radar screen"
        ],
        "LISTRTE": [
            "LISTRTE acid, [pagenr]",
            "acid,[int]",
            lambda idx, *args: bs.traf.ap.route[idx].listrte(idx, *args),
            "Show list of route in window per page of 5 waypoints"
        ],
        "LNAV": [
            "LNAV acid,[ON/OFF]",
            "acid,[onoff]",
            bs.traf.ap.setLNAV,
            "LNAV (lateral FMS mode) switch for autopilot"
        ],
        "MAKEDOC": [
            "MAKEDOC",
            "",
            makedoc,
            "Make markdown templates for all stack functions that don't have a doc page yet."
        ],
        "MCRE": [
            "MCRE n, [type/*, alt/*, spd/*, dest/*]",
            "int,[txt,alt,spd,txt]",
            bs.traf.mcreate,
            "Multiple random create of n aircraft in current view"
        ],
        # "METRIC": [
        #     "METRIC OFF/0/1/2, [dt]",
        #     "onoff/int,[float]",
        #     bs.sim.metric.toggle,
        #     "Complexity metrics module"
        # ],
        "MOVE": [
            "MOVE acid,lat,lon,[alt,hdg,spd,vspd]",
            "acid,latlon,[alt,hdg,spd,vspd]",
            bs.traf.move,
            "Move an aircraft to a new position"
        ],
        "ND": [
            "ND acid",
            "txt",
            lambda acid: bs.scr.feature("ND", acid),
            "Show navigation display with CDTI"
        ],
        "NOISE": [
            "NOISE [ON/OFF]",
            "[onoff]",
            bs.traf.setNoise,
            "Turbulence/noise switch"
        ],
        "NOM": [
            "NOM acid",
            "acid",
            bs.traf.nom,
            "Set nominal acceleration for this aircraft (perf model)"
        ],
        "NORESO": [
            "NORESO [acid]",
            "[string]",
            bs.traf.asas.SetNoreso,
            "Switch off conflict resolution for this aircraft"
        ],
        "OP": [
            "OP",
            "",
            bs.sim.start,
            "Start/Run simulation or continue after pause"
        ],
        "ORIG": [
            "ORIG acid, latlon/airport",
            "acid,wpt/latlon",
            lambda *args: bs.traf.ap.setdestorig("ORIG", *args),
            "Set origin airport of aircraft"
        ],
        "PAN": [
            "PAN latlon/acid/airport/waypoint/LEFT/RIGHT/ABOVE/DOWN",
            "pandir/latlon",
            bs.scr.pan,
            "Pan screen (move view) to a waypoint, direction or aircraft"
        ],
        "PCALL": [
            "PCALL filename [REL/ABS]",
            "txt,[txt]",
            lambda *args: openfile(*args, mergeWithExisting=True),
            "Call commands in another scenario file"
        ],
        "PLUGINS": [
            "PLUGINS LIST or LOAD plugin or REMOVE plugin",
            "txt,[txt]",
            plugin.manage,
            "List all plugins, load a plugin, or remove a loaded plugin."
        ],
        "POLY": [
            "POLY name,lat,lon,lat,lon, ...",
            "txt,latlon,...",
            lambda name, *coords: areafilter.defineArea(name, 'POLY', coords),
            "Define a polygon-shaped area"
        ],
        "POLYALT": [
            "POLYALT name,top,bottom,lat,lon,lat,lon, ...",
            "txt,alt,alt,latlon,...",
            lambda name, top, bottom, *coords: areafilter.defineArea(name, 'POLYALT', coords, top, bottom),
            "Define a polygon-shaped area in 3D: between two altitudes"
        ],
        "POS": [
            "POS acid/waypoint",
            "acid/wpt",
            bs.traf.poscommand,
            "Get info on aircraft, airport or waypoint"
        ],
        "PRIORULES": [
            "PRIORULES [ON/OFF PRIOCODE]",
            "[onoff, txt]",
            bs.traf.asas.SetPrio,
            "Define priority rules (right of way) for conflict resolution"
        ],
        "QUIT": [
            "QUIT",
            "",
            bs.sim.stop,
            "Quit program/Stop simulation"
        ],
        "RESET": [
            "RESET",
            "",
            bs.sim.reset,
            "Reset simulation"
        ],
        "RFACH": [
            "RFACH [factor]",
            "[float]",
            bs.traf.asas.SetResoFacH,
            "Set resolution factor horizontal (to add a margin)"
        ],
        "RFACV": [
            "RFACV [factor]",
            "[float]",
            bs.traf.asas.SetResoFacV,
            "Set resolution factor vertical (to add a margin)"
        ],
        "RESO": [
            "RESO [method]",
            "[txt]",
            bs.traf.asas.SetCRmethod,
            "Set resolution method"
        ],
        "RESOOFF": [
            "RESOOFF [acid]",
            "[string]",
            bs.traf.asas.SetResooff,
            "Switch for conflict resolution module"
        ],
        "RMETHH": [
            "RMETHH [method]",
            "[txt]",
            bs.traf.asas.SetResoHoriz,
            "Set resolution method to be used horizontally"
        ],
        "RMETHV": [
            "RMETHV [method]",
            "[txt]",
            bs.traf.asas.SetResoVert,
            "Set resolution method to be used vertically"
        ],
        "RSZONEDH": [
            "RSZONEDH [height]",
            "[float]",
            bs.traf.asas.SetPZHm,
            "Set half of vertical dimension of resolution zone in ft"
        ],
        "RSZONER": [
            "RSZONER [radius]",
            "[float]",
            bs.traf.asas.SetPZRm,
            "Set horizontal radius of resolution zone in nm"
        ],
        "SAVEIC": [
            "SAVEIC filename",
            "string",
            saveic,
            "Save current situation as IC"
        ],
        "SCHEDULE": [
            "SCHEDULE time, COMMAND+ARGS",
            "time,txt,...",
            lambda time, *args: sched_cmd(time, args, relative=False),
            "Schedule a stack command at a given time"
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
            bs.traf.ap.selspdcmd,
            "Speed command (autopilot)"
        ],
        "SSD": [
            "SSD ALL/CONFLICTS/OFF or SSD acid0, acid1, ...",
            "txt,[...]",
            bs.scr.showssd,
            "Show state-space diagram (=conflict prevention display/predictive ASAS)"
        ],
        "SWRAD": [
            "SWRAD GEO/GRID/APT/VOR/WPT/LABEL/ADSBCOVERAGE/TRAIL [dt]/[value]",
            "txt,[float]",
            bs.scr.feature,
            "Switch on/off elements and background of map/radar view"
        ],
        "SYMBOL": [
            "SYMBOL",
            "",
            bs.scr.symbol,
            "Toggle aircraft symbol"
        ],
        "SYN": [
            " SYN: Possible subcommands: HELP, SIMPLE, SIMPLED, DIFG, SUPER," + \
            "MATRIX, FLOOR, TAKEOVER, WALL, ROW, COLUMN, DISP",
            "txt,[...]",
            syn.process,
            "Macro for generating synthetic (geometric) traffic scenarios"
        ],
        "TIME": [
            "TIME RUN(default) / HH:MM:SS.hh / REAL / UTC ",
            "[txt]",
            bs.sim.setclock,
            "Set simulated clock time"
        ],
        "TRAIL": [
            "TRAIL ON/OFF, [dt] OR TRAIL acid color",
            "[acid/bool],[float/txt]",
            bs.traf.trails.setTrails,
            "Toggle aircraft trails on/off"
        ],
        "VNAV": [
            "VNAV acid,[ON/OFF]",
            "acid,[onoff]",
            bs.traf.ap.setVNAV,
            "Switch on/off VNAV mode, the vertical FMS mode (autopilot)"
        ],
        "VS": [
            "VS acid,vspd (ft/min)",
            "acid,vspd",
            bs.traf.ap.selvspdcmd,
            "Vertical speed command (autopilot)"
        ],
        "WIND": [
            "WIND lat,lon,alt/*,dir,spd,[alt,dir,spd,alt,...]",
            "latlon,[alt],float,float,...,...,...",   # last 3 args are repeated
            bs.traf.wind.add,
            "Define a wind vector as part of the 2D or 3D wind field"
        ],
        "ZONEDH": [
            "ZONEDH [height]",
            "[float]",
            bs.traf.asas.SetPZH,
            "Set half of the vertical protected zone dimensions in ft"
        ],
        "ZONER": [
            "ZONER [radius]",
            "[float]",
            bs.traf.asas.SetPZR,
            "Set the radius of the horizontal protected zone dimensions in nm"

        ],
        "ZOOM": [
            "ZOOM IN/OUT or factor",
            "float/txt",
            lambda a: bs.scr.zoom(1.4142135623730951) if a == "IN" else \
                      bs.scr.zoom(0.7071067811865475) if a == "OUT" else \
                      bs.scr.zoom(a, True),
            "Zoom display in/out, you can also use +++ or -----"
        ]
    }

    cmddict.update(commands)

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

def remove_commands(commands):
    """ Remove functions from the stack """
    for cmd in commands:
        cmddict.pop(cmd)

def showhelp(cmd=''):
    """ Generate help text for displaying or dump command reference in file
        when command is >filename
    """
    # No command given: show all
    if len(cmd) == 0:
        return "There are different ways to get help:\n" + \
               " HELP PDF  gives an overview of the existing commands\n" + \
               " HELP cmd  gives a help line on the command (syntax)\n"  + \
               " DOC  cmd  show documentation of a command (if available)\n" + \
               "And there is more info in the docs folder and the wiki on Github"

    elif cmd.upper() == "PDF":
        os.chdir("docs")
        pdfhelp = "BLUESKY-COMMAND-TABLE.pdf"
        if os.path.isfile(pdfhelp):
            try:
                subprocess.Popen(pdfhelp, shell=True)
            except:
                return "Opening " + pdfhelp + " failed."
        else:
            return pdfhelp + "does not exist."
        os.chdir("..")
        return "Pdf window opened"

    # Show help line for command
    elif cmd in cmddict:

        # Check whether description is available, then show it as well
        if len(cmddict) <= 3:
            return cmddict[cmd][0]
        else:
            return cmddict[cmd][0] + "\n" + cmddict[cmd][3]

    # Show help line for equivalent command
    elif cmd in cmdsynon:

        # Check whether description is available, then show it as well
        if len(cmddict[cmdsynon[cmd]]) <= 3:
            return cmddict[cmdsynon[cmd]][0]
        else:
            return cmddict[cmdsynon[cmd]][0] + "\n" + cmddict[cmdsynon[cmd]][3]

    # Write command reference to tab-delimited text file
    elif cmd[0] == ">":

        # Get filename
        if len(cmd) > 1:
            fname = "./docs/" + cmd[1:]
        else:
            fname = "./docs/bluesky-commands.txt"

        # Write command dictionary to tab-delimited text file
        try:
            f = open(fname, "w")
        except:
            return "Invalid filename:" + fname

        # Header of first table
        f.write("Command\tDescription\tUsage\tArgument types\tFunction\n")

        table = []  # for alphabetical sort use a table

        # Get info for all commands
        for item, lst in cmddict.items():
            line = item + "\t"
            if len(lst) > 3:
                line = line + lst[3]
            line = line + "\t" + lst[0] + "\t" + str(lst[1]) + "\t"

            # Clean up string with function name and add if not a lambda function
            funct = str(lst[2]).replace("<", "").replace(">", "")

            # Lambda function give no info, also remove hex address and "method" text
            if funct.count("lambda") == 0:

                if funct.count("at") > 0:
                    idxat = funct.index(" at ")
                    funct = funct[:idxat]

                funct = funct.replace("bound method", "")
                line = line + funct

            table.append(line)

        # Sort & write table
        table.sort()
        for line in table:
            f.write(line + "\n")
        f.write("\n")

        # Add synonyms table
        f.write("\n\n Synonyms (equivalent commands)\n")

        table = []  # for alphabetical sort use table
        for item in cmdsynon:
            if cmdsynon[item] in cmddict and len(cmddict[cmdsynon[item]]) >= 3:
                table.append(item + "\t" + cmdsynon[item] + "\t" + cmddict[cmdsynon[item]][3])
            else:
                table.append(item + "\t" + cmdsynon[item] + "\t")

        # Sort & write table
        table.sort()
        for line in table:
            f.write(line + "\n")
        f.write("\n")

        # Close and report where file is to be found
        f.close()
        return "Writing command reference in " + fname

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


def stack(cmdline, sender_id=None):
    # Stack one or more commands separated by ";"
    cmdline = cmdline.strip()
    if len(cmdline) > 0:
        for line in cmdline.split(';'):
            cmdstack.append((line, sender_id))


def sched_cmd(time, args, relative=False):
    tostack = ','.join(args)
    # find spot in time list corresponding to passed time, get idx
    # insert time at idx in scentime, insert cmd at idx in scencmd
    if relative:
        time += bs.sim.simt
    # in case there is no scentime yet, only extend

    if len(scentime) == 0:
        scentime.extend([time])
        scencmd.extend([tostack])
    else:
        try:
            idx = scentime.index(next(sctime for sctime in scentime if sctime > time))

            scentime.insert(idx, time)
            scencmd.insert(idx, tostack)
        except:
            scentime.extend([time])
            scencmd.extend([tostack])


    return True


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

    print("Opening "+scenfile)

    # If timestamps in file should be interpreted as relative we need to add
    # the current simtime to every timestamp
    t_offset = bs.sim.simt if absrel == 'REL' else 0.0

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
                        print("except this:"+line)
                    pass  # nice try, we will just ignore this syntax error

    if mergeWithExisting:
        # If we are merging we need to sort the resulting command list
        scentime, scencmd = [list(x) for x in zip(*sorted(
            zip(scentime, scencmd), key=lambda pair: pair[0]))]

    return True


def ic(filename=''):
    global scenfile, scenname

    # Get the filename of new scenario
    if filename == '':
        filename = bs.scr.show_file_dialog()

    # Clean up filename
    filename = filename.strip()

    # Reset sim and open new scenario file
    if len(filename) > 0:
        bs.sim.reset()
        result = openfile(filename)

        if result is True:
            scenfile    = filename
            scenname, _ = os.path.splitext(os.path.basename(filename))
            # Remember this filename in IC.scn in scenario folder
            keepicfile = open(settings.scenario_path+"/"+"ic.scn","w")
            keepicfile.write("# This file is used by BlueSky to save the last used scenario file\n")
            keepicfile.write("# So in the console type 'IC IC' to restart the previously used scenario file\n")
            keepicfile.write("00:00:00.00>IC "+filename+"\n")
            keepicfile.close()
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


def saveic(fname):
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

    for i in range(bs.traf.nbs.traf):
        # CRE acid,type,lat,lon,hdg,alt,spd
        cmdline = "CRE " + bs.traf.id[i] + "," + bs.traf.type[i] + "," + \
                  repr(bs.traf.lat[i]) + "," + repr(bs.traf.lon[i]) + "," + \
                  repr(bs.traf.trk[i]) + "," + repr(bs.traf.alt[i] / ft) + "," + \
                  repr(tas2cas(bs.traf.tas[i], bs.traf.alt[i]) / kts)

        f.write(timtxt + cmdline + "\n")

        # VS acid,vs
        if abs(bs.traf.vs[i]) > 0.05:  # 10 fpm dead band
            if abs(bs.traf.ap.vs[i]) > 0.05:
                vs_ = bs.traf.ap.vs[i] / fpm
            else:
                vs_ = bs.traf.vs[i] / fpm

            cmdline = "VS " + bs.traf.id[i] + "," + repr(vs_)
            f.write(timtxt + cmdline + "\n")

        # Autopilot commands
        # Altitude
        if abs(bs.traf.alt[i] - bs.traf.ap.alt[i]) > 10.:
            cmdline = "ALT " + bs.traf.id[i] + "," + repr(bs.traf.ap.alt[i] / ft)
            f.write(timtxt + cmdline + "\n")

        # Heading as well when heading select
        delhdg = (bs.traf.hdg[i] - bs.traf.ap.trk[i] + 180.) % 360. - 180.
        if abs(delhdg) > 0.5:
            cmdline = "HDG " + bs.traf.id[i] + "," + repr(bs.traf.ap.trk[i])
            f.write(timtxt + cmdline + "\n")

        # Speed select? => Record
        rho = density(bs.traf.alt[i])  # alt in m!
        aptas = sqrt(1.225 / rho) * bs.traf.ap.spd[i]
        delspd = aptas - bs.traf.tas[i]

        if abs(delspd) > 0.4:
            cmdline = "SPD " + bs.traf.id[i] + "," + repr(bs.traf.ap.spd[i] / kts)
            f.write(timtxt + cmdline + "\n")

        # DEST acid,dest-apt
        if bs.traf.ap.dest[i] != "":
            cmdline = "DEST " + bs.traf.id[i] + "," + bs.traf.ap.dest[i]
            f.write(timtxt + cmdline + "\n")

        # ORIG acid,orig-apt
        if bs.traf.ap.orig[i] != "":
            cmdline = "ORIG " + bs.traf.id[i] + "," + bs.traf.ap.orig[i]
            f.write(timtxt + cmdline + "\n")

        # Route with ADDWPT
        route = bs.traf.ap.route[i]
        for iwp in range(route.nwp):
            # dets and orig al already done, skip them here
            if iwp == 0 and route.wpname[iwp] == bs.traf.ap.orig[i]:
                continue

            if iwp == route.nwp - 1 and route.wpname[iwp] == bs.traf.ap.dest[i]:
                continue

            #add other waypoints
            cmdline = "ADDWPT " + bs.traf.id[i] + " "
            wpname = route.wpname[iwp]
            if wpname[:len(bs.traf.id[i])] == bs.traf.id[i]:
                wpname = repr(route.lat[iwp]) + "," + repr(route.lon[iwp])
            cmdline = cmdline + wpname + ","

            if route.wpalt[iwp] >= 0.:
                cmdline = cmdline + repr(route.wpalt[iwp] / ft) + ","
            else:
                cmdline = cmdline + ","

            if route.wpspd[iwp] >= 0.:
                if route.wpspd[iwp] > 1.0:
                    cmdline = cmdline + repr(route.wpspd[iwp] / kts)
                else:
                    cmdline = cmdline + repr(route.wpspd[iwp])

            f.write(timtxt + cmdline + "\n")

    # Saveic: should close
    f.close()
    return True


def process():
    """process and empty command stack"""

    # Process stack of commands
    for (line, sender_id) in cmdstack:
        #debug       print "stack is processing:",line
        # Empty line: next command
        line = line.strip()
        if len(line) == 0:
            continue

        # Split command line into command and arguments, pass traf ids to check for
        # switched acid and command
        cmd, args = cmdsplit(line.upper(), bs.traf.id)
        numargs   = len(args)
        # Check if this is a POS command with only an aircraft id
        if numargs == 0 and bs.traf.id.count(cmd) > 0:
            args    = [cmd]
            cmd     = 'POS'
            numargs = 1

        # Assume syntax is ok (default)
        synerr = False

        #**********************************************************************
        #=====================  Start of command parsing  =====================
        #**********************************************************************

        #----------------------------------------------------------------------
        # First check command synonyms list, then in dictionary
        #----------------------------------------------------------------------
        orgcmd = cmd  # save for string cutting out of line and use of synonyms
        if cmd in list(cmdsynon.keys()):
            cmd    = cmdsynon[cmd]

        if cmd in list(cmddict.keys()):
            # Look up command in dictionary to get string with argtypes andhelp texts
            helptext, argtypelist, function = cmddict[cmd][:3]

            # Make list of argtypes and whether entering an argument is optional
            argtypes = []
            argisopt = []

            # Process and reduce arglist from left to right
            # First cut at square brackets, then take separate argument types
            while len(argtypelist) > 0:
                opt = (argtypelist[0] == '[')
                cut = argtypelist.find(']') if opt else \
                      argtypelist.find('[') if '[' in argtypelist else \
                      len(argtypelist)

                types = argtypelist[:cut].strip('[,]').split(',')
                argtypes += types
                argisopt += len(types) * [opt]
                argtypelist = argtypelist[cut:].lstrip(',]')

            # Check if at least the number of mandatory arguments is given,
            # by finding the last argument that is not optional.
            if False in argisopt:
                minargs = len(argisopt) - argisopt[::-1].index(False)
                if numargs < minargs:
                    bs.scr.echo("Syntax error: Too few arguments", sender_id)
                    bs.scr.echo(line, sender_id)
                    bs.scr.echo(helptext, sender_id)
                    continue

            # Special case: single text string argument: case sensitive,
            # possibly with spaces/newlines pass the original
            if argtypes == ['string']:
                arglist = [line[len(orgcmd) + 1:]]

            else:
                # Start with a fresh argument parser for each command
                parser  = Argparser()
                arglist = []
                curtype = 0
                curarg  = 0

                # Iterate over list of argument types & arguments
                while curtype < len(argtypes) and curarg < len(args) and not synerr:
                    # Optional repeat with "...", e.g. for lat/lon list for polygon
                    if argtypes[curtype][:3] == '...':
                        repeatsize = len(argtypes) - curtype
                        curtype = curtype - repeatsize
                    argtype    = argtypes[curtype].strip().split('/')

                    # Save error messages from argument parsing for each possible type for this field
                    errors = ''
                    # Go over all argtypes separated by "/" in this place in the command line
                    for i, argtypei in enumerate(argtype):
                        # Try to parse the argument for the given argument type
                        # First successful parsing is used!
                        if parser.parse(argtypei, curarg, args):
                            # No value = None when this is allowed because it is an optional argument
                            if parser.result[0] is None and argisopt[curtype] is False:
                                synerr = True
                                bs.scr.echo('No value given for mandatory argument ' + argtypes[curtype], sender_id)
                                break
                            arglist += parser.result
                            curarg  += parser.argstep
                            break
                        # No success yet with this type (maybe we can try other ones)
                        else:
                            # Store the error message and see if there are alternatives
                            errors += parser.error + '\n'
                            if i < len(argtype) - 1:
                                # We have alternative argument formats that we can try
                                continue
                            else:
                                # No more types to check: print error message
                                synerr = True
                                bs.scr.echo('Syntax error processing "' + args[curarg] + '":', sender_id)
                                bs.scr.echo(errors, sender_id)
                                bs.scr.echo(helptext, sender_id)
                                print("Error in processing arguments:")
                                print(line)

                    curtype += 1

            # Call function return flag,text
            # flag: indicates sucess
            # text: optional error message
            if not synerr:
                results = function(*arglist)  # * = unpack list to call arguments
                if isinstance(results, bool):  # Only flag is returned
                    synerr = not results
                    if synerr:
                        if numargs <= 0 or curarg < len(args) and args[curarg] == "?":
                            bs.scr.echo(helptext, sender_id)
                        else:
                            bs.scr.echo("Syntax error: " + helptext, sender_id)

                elif isinstance(results, tuple) and len(results) > 0:
                    synerr = not results[0]
                    if synerr:
                        bs.scr.echo("Syntax error: " + (helptext if len(results) < 2 else ""), sender_id)
                    # Maybe there is also an error/info message returned?
                    if len(results) >= 2:
                        prefix = "" if results[0] == bs.SIMPLE_ECHO \
                            else "{}: ".format(cmd)
                        bs.scr.echo("{}{}".format(prefix, results[1]), sender_id)

                bs.scr.echo(bs.MSG_OK, sender_id)

            else:  # synerr:
                bs.scr.echo("Syntax error: " + helptext, sender_id)

        #----------------------------------------------------------------------
        # ZOOM command (or use ++++  or --  to zoom in or out)
        #----------------------------------------------------------------------
        elif cmd[0] in ["+", "=", "-"]:
            nplus = cmd.count("+") + cmd.count("=")  # = equals + (same key)
            nmin  = cmd.count("-")
            bs.scr.zoom(sqrt(2) ** (nplus - nmin), absolute=False)

        #-------------------------------------------------------------------
        # Command not found
        #-------------------------------------------------------------------
        else:
            if numargs == 0:
                bs.scr.echo("Unknown command or aircraft: " + cmd, sender_id)
            else:
                bs.scr.echo("Unknown command: " + cmd, sender_id)

        #**********************************************************************
        #======================  End of command branches ======================
        #**********************************************************************

    # End of for-loop of cmdstack
    del cmdstack[:]
    return


class Argparser:
    # Global variables
    reflat    = -999.  # Reference latitude for searching in nav db
                       # in case of duplicate names
    reflon    = -999.  # Reference longitude for searching in nav db
                       # in case of duplicate names

    def __init__(self):
        self.result     = []  # The outcome of a parsed argument is stored here
        self.argstep    = 0   # The number of arguments that were parsed
        self.error      = ''  # Potential parsing error messages are stored here
        self.additional = {}  # Sometimes a parse can generate extra arguments
                              # that can be used to fill future empty arguments.
                              # E.g., a runway gives a lat/lon, but also a heading.
        self.refac      = -1  # Stored aircraft idx when an argument is parsed
                              # for a function that acts on an aircraft.

    def parse(self, argtype, argidx, args):
        """ Parse one or more arguments.

            Returns True if parse was successful. When not successful, False is
            returned, and the error message is stored in self.error """

        # First reset outcome values
        self.result  = []
        self.argstep = 0
        self.error   = ''

        if argtype == "txt":  # simple text
            self.result  = [args[argidx]]
            self.argstep = 1
            return True

        # Empty arg or wildcard
        elif args[argidx] == "" or args[argidx] == "*":
            # If there was a matching additional argument stored previously use that one
            if argtype in self.additional and args[argidx] == "*":
                self.result  = [self.additional[argtype]]
                self.argstep = 1
                return True
            # Otherwise result is None
            self.result  = [None]
            self.argstep = 1
            return True

        if argtype == "acid":  # aircraft id => parse index
            idx = bs.traf.id2idx(args[argidx])
            if idx < 0:
                self.error = args[argidx] + " not found"
                return False
            else:
                # Update ref position for navdb lookup
                Argparser.reflat = bs.traf.lat[idx]
                Argparser.reflon = bs.traf.lon[idx]
                self.refac   = idx
                self.result  = [idx]
                self.argstep = 1
                return True

        elif argtype == "wpinroute":  # return text in upper case
            wpname = args[argidx].upper()
            if self.refac >= 0 and wpname not in bs.traf.ap.route[self.refac].wpname:
                self.error = 'There is no waypoint ' + wpname + ' in route of ' + bs.traf.id[self.refac]
                return False
            self.result  = [wpname]
            self.argstep = 1
            return True

        elif argtype == "float":  # float number
            try:
                self.result  = [float(args[argidx])]
                self.argstep = 1
                return True
            except ValueError:
                self.error = 'Argument "' + args[argidx] + '" is not a float'
                return False

        elif argtype == "int":   # integer
            try:
                self.result  = [int(args[argidx])]
                self.argstep = 1
                return True
            except ValueError:
                self.error = 'Argument "' + args[argidx] + '" is not an int'
                return False

        elif argtype == "onoff" or argtype == "bool":
            if args[argidx] in ["ON", "TRUE", "YES", "1"]:
                self.result  = [True]
                self.argstep = 1
                return True
            elif args[argidx] in ["OFF", "FALSE", "NO", "0"]:
                self.result  = [False]
                self.argstep = 1
                return True
            else:
                self.error = 'Argument "' + args[argidx] + '" is not a bool'
                return False

        elif argtype == "wpt" or argtype == "latlon":

            # wpt: Make 1 or 2 argument(s) into 1 position text to be used as waypoint
            # latlon: return lat,lon to be used as a position only

            # Examples valid position texts:
            # lat/lon : "N52.12,E004.23","N52'14'12',E004'23'10"
            # navaid/fix: "SPY","OA","SUGOL"
            # airport:   "EHAM"
            # runway:    "EHAM/RW06" "LFPG/RWY23"
            # Default values
            self.argstep = 1
            name         = args[argidx]

            # Try aircraft first: translate a/c id into a valid position text with a lat,lon
            idx = bs.traf.id2idx(name)
            if idx >= 0:
                name     = str(bs.traf.lat[idx]) + "," + str(bs.traf.lon[idx])

            # Check next arg if available
            elif argidx < len(args) - 1:
                # lat,lon ? Combine into one string with a comma
                if islat(args[argidx]):
                    name = args[argidx] + "," + args[argidx + 1]
                    self.argstep = 2   # we used two arguments

                # apt,runway ? Combine into one string with a slash as separator
                elif args[argidx + 1][:2].upper() == "RW" and args[argidx] in bs.navdb.aptid:
                    name = args[argidx] + "/" + args[argidx + 1].upper()
                    self.argstep = 2   # we used two arguments

            # Return something different for the two argtypes:

            # wpt argument type: simply return positiontext, no need it look up nw
            if argtype == "wpt":
                self.result = [name]
                return True

            # lat/lon argument type we also need to it up:
            elif argtype == "latlon":
                # Set default reference lat,lon for duplicate name in navdb to screen
                if Argparser.reflat < -180.:  # No reference avaiable yet: use screen center
                    Argparser.reflat = bs.scr.ctrlat
                    Argparser.reflon = bs.scr.ctrlon

                success, posobj = txt2pos(name, Argparser.reflat, Argparser.reflon)

                if success:
                    # for runway type, get heading as default optional argument for command line
                    if posobj.type == "rwy":
                        aptname, rwyname = name.split('/RW')
                        rwyname = rwyname.lstrip('Y')
                        try:
                            self.additional['hdg'] = bs.navdb.rwythresholds[aptname][rwyname][2]
                        except:
                            pass

                    # Update reference lat/lon
                    Argparser.reflat = posobj.lat
                    Argparser.reflon = posobj.lon

                    self.result = [posobj.lat, posobj.lon]
                    return True
                else:
                    self.error = posobj  # contains error message if txt2pos was no success
                    return False

        # Pan direction: check for valid string value
        elif argtype == "pandir":
            pandir = args[argidx].upper().strip()
            if pandir in ["LEFT", "RIGHT", "UP", "ABOVE", "RIGHT", "DOWN"]:
                self.result  = pandir
                self.argstep = 1
                return True
            else:
                self.error = pandir + ' is not a valid pan argument'
                return False

        # CAS[kts] Mach: convert kts to m/s for values=>1.0 (meaning CAS)
        elif argtype == "spd":

            try:
                spd = float(args[argidx].upper()
                       .replace("M0.", ".").replace("M", ".").replace("..", "."))

                if not (0.1 < spd < 1.0 or args[argidx].count("M") > 0):
                    spd = spd * kts
                self.result  = [spd]
                self.argstep = 1
                return True
            except ValueError:
                self.error = 'Could not parse "' + args[argidx] + '" as speed'
                return False

        # Vertical speed: convert fpm to in m/s
        elif argtype == "vspd":
            try:
                self.result  = [fpm * float(args[argidx])]
                self.argstep = 1
                return True
            except ValueError:
                self.error = 'Could not parse "' + args[argidx] + '" as vertical speed'
                return False

        # Altutide convert ft or FL to m
        elif argtype == "alt":  # alt: FL250 or 25000 [ft]
            alt = txt2alt(args[argidx])

            if alt > -1e8:
                self.result  = [alt * ft]
                self.argstep = 1
                return True
            else:
                self.error = 'Could not parse "' + args[argidx] + '" as altitude'
                return False

        # Heading: return float in degrees
        elif argtype == "hdg":
            try:
                # TODO: take care of difference between magnetic/true heading
                hdg = float(args[argidx].upper().replace('T', '').replace('M', ''))
                self.result  = [hdg]
                self.argstep = 1
                return True
            except ValueError:
                self.error = 'Could not parse "' + args[argidx] + '" as heading'
                return False

        # Time: convert time MM:SS.hh or HH:MM:SS.hh to a float in seconds
        elif argtype == "time":
            try:
                ttxt = args[argidx].strip().split(':')
                if len(ttxt) >= 3:
                    ihr  = int(ttxt[0]) * 3600.0
                    imin = int(ttxt[1]) * 60.0
                    xsec = float(ttxt[2])
                    self.result = [ihr + imin + xsec]
                else:
                    self.result = [float(args[argidx])]
                self.argstep = 1
                return True
            except ValueError:
                self.error = 'Could not parse "' + args[argidx] + '" as time'
                return False

        # Argument not found: return False
        self.error = 'Unknown argument type: ' + argtype
        return False


def distcalc(lat0, lon0, lat1, lon1):
    try:
        qdr, dist = geo.qdrdist(lat0, lon0, lat1, lon1)
        return bs.SIMPLE_ECHO, "QDR = %.2f deg, Dist = %.3f nm" % (qdr % 360., dist)
    except:
        return False, 'Error in dist calculation.'


def makedoc():
    import re
    re_args = re.compile(r'\w+')
    if not os.path.isdir('tmp'):
        os.mkdir('tmp')
    for name, lst in cmddict.items():
        if not os.path.isfile('data/html/%s.html' % name.lower()):
            with open('tmp/%s.md' % name.lower(), 'w') as f:
                f.write('# %s: %s\n' % (name, name.capitalize()) +
                    lst[3] + '\n\n' +
                    '**Usage:**\n\n' +
                    '    %s\n\n' % lst[0] +
                    '**Arguments:**\n\n')
                if len(lst[1]) == 0:
                    f.write('This command has no arguments.\n\n')
                else:
                    f.write('|Name|Type|Optional|Description\n' +
                        '|--------|------|---|---------------------------------------------------\n')
                    for arg in re_args.findall(lst[0])[1:]:
                        f.write(arg + '|     |   |\n')
                f.write('\n[[Back to command reference.|Command Reference]]\n')
