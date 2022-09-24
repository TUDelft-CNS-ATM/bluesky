''' BlueSky base stack commands. '''
import webbrowser
from pathlib import Path

import bluesky as bs
from bluesky import settings
from bluesky.core import select_implementation, simtime, varexplorer as ve
from bluesky.tools import geo, aero, areafilter, plotter
from bluesky.tools.calculator import calculator
from bluesky.stack.cmdparser import append_commands


def initbasecmds():
    ''' Initialise BlueSky base stack commands. '''
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
    #   hdg       = heading in degrees, True or Magnetic
    #
    #   float     = plain float
    #   int       = integer
    #   txt       = text will be converted to upper case
    #               (for keywords, navaids, flags, waypoints,acid etc)
    #   word      = single, case sensitive word
    #   string    = case sensitive string
    #   on/off    = text => boolean
    #
    #   latlon    = converts acid, wpt, airport etc => lat,lon (deg) so 2 args!
    #   wpt       = converts postext or lat,lon into a text string,
    #               to be used as named waypoint
    #   wpinroute = text string with name of waypoint in route
    #   pandir    = text with LEFT, RIGHT, UP/ABOVE or DOWN
    #
    # Below this dictionary also a dictionary of synonym commands is given
    #
    # --------------------------------------------------------------------
    cmddict = {
        "ADDNODES": [
            "ADDNODES number",
            "int",
            bs.net.addnodes,
            "Add a simulation instance/node",
        ],
        "AIRWAY": [
            "AIRWAY wp/airway",
            "txt",
            bs.traf.airwaycmd,
            "Get info on airway or connections of a waypoint",
        ],
        "ATALT": [
            "acid ATALT alt cmd ",
            "acid,alt,string",
            bs.traf.cond.ataltcmd,
            "When a/c at given altitude , execute a command cmd",
        ],
        "ATDIST": [
            "acid ATDIST pos dist cmd ",
            "acid,latlon,float,string",
            bs.traf.cond.atdistcmd,
            "When a/c passing this distance[nm] to position, execute the command cmd",
        ],
        "ATSPD": [
            "acid ATSPD spd cmd ",
            "acid,spd,string",
            bs.traf.cond.atspdcmd,
            "When a/c reaches given speed, execute a command cmd",
        ],
        "BANK": [
            "BANK acid bankangle[deg]",
            "acid,[float]",
            bs.traf.setbanklim,
            "Set or show bank limit for this vehicle",
        ],
        "BATCH": [
            "BATCH filename",
            "string",
            bs.sim.batch,
            "Start a scenario file as batch simulation",
        ],
        "BLUESKY": ["BLUESKY", "", singbluesky, "Sing"],
        "BENCHMARK": [
            "BENCHMARK [scenfile,time]",
            "[string,time]",
            bs.sim.benchmark,
            "Run benchmark",
        ],
        "BOX": [
            "BOX name,lat,lon,lat,lon,[top,bottom]",
            "txt,latlon,latlon,[alt,alt]",
            lambda name, *coords: areafilter.defineArea(
                name, "BOX", coords[:4], *coords[4:]
            ),
            "Define a box-shaped area",
        ],
        "CALC": [
            "CALC expression",
            "string",
            calculator,
            "Simple in-line math calculator, evaluates expression",
        ],
        "CASMACHTHR": [
            "CASMACHTHR threshold",
            "float",
            aero.casmachthr,
            """Set a threshold below which speeds should be considered as Mach numbers
               in CRE(ATE), ADDWPT, and SPD commands. Set to zero if speeds should
               never be considered as Mach number(e.g., when simulating drones)."""
        ],
        "CD": [
            "CD [path]",
            "[word]",
            setscenpath,
            "Change to a different scenario folder",
        ],
        "CIRCLE": [
            "CIRCLE name,lat,lon,radius,[top,bottom]",
            "txt,latlon,float,[alt,alt]",
            lambda name, *coords: areafilter.defineArea(
                name, "CIRCLE", coords[:3], *coords[3:]
            ),
            "Define a circle-shaped area",
        ],
        "CLRCRECMD": [
            "CLRCRECMD",
            "",
            bs.traf.clrcrecmd,
            "CLRCRECMD will clear CRECMD list of commands a/c creation",
        ],
        "COLOR": [
            "COLOR name,color (named color or r,g,b)",
            "txt,color",
            bs.scr.color,
            "Set a custom color for an aircraft or shape",

        ],
        "CRE": [
            "CRE acid,type,lat,lon,hdg,alt,spd",
            "txt,txt,latlon,[hdg,alt,spd]",
            bs.traf.cre,
            "Create an aircraft",
        ],
        "CRECMD": [
            "CRECMD cmdline (to be added after a/c id )",
            "string",
            bs.traf.crecmd,
            "Add a command for each aircraft to be issued after creation of aircraft",
        ],
        "CRECONFS": [
            "CRECONFS id, type, targetid, dpsi, cpa, tlos_hor, dH, tlos_ver, spd",
            "txt,txt,acid,hdg,float,time,[alt,time,spd]",
            bs.traf.creconfs,
            "Create an aircraft that is in conflict with 'targetid'",
        ],
        "DATE": [
            "DATE [day,month,year,HH:MM:SS.hh]",
            "[int,int,int,txt]",
            bs.sim.setutc,
            "Set simulation date",
        ],
        "DEFWPT": [
            "DEFWPT wpname,lat,lon,[FIX/VOR/DME/NDB]",
            "txt,latlon,[txt]",
            bs.navdb.defwpt,
            "Define a waypoint only for this scenario/run",
        ],
        "DEL": [
            "DEL acid/ALL/WIND/shape",
            "acid/txt,...",
            lambda *a: bs.traf.wind.clear()
            if isinstance(a[0], str) and a[0] == "WIND"
            else areafilter.deleteArea(a[0])
            if isinstance(a[0], str)
            else bs.traf.groups.delgroup(a[0])
            if hasattr(a[0], "groupname")
            else bs.traf.delete(a),
            "Delete command (aircraft, wind, area)",
        ],

        "DIST": [
            "DIST lat0, lon0, lat1, lon1",
            "latlon,latlon",
            distcalc,
            "Distance and direction calculation between two positions",
        ],
        "DOC": [
            "DOC [command]",
            "[txt]",
            bs.scr.show_cmd_doc,
            "Show extended help window for given command, or the main documentation page if no command is given.",
        ],
        "DT": [
            "DT [dt] OR [target,dt]",
            "[float/txt,float]",
            lambda *args: simtime.setdt(*reversed(args)),
            "Set simulation time step",
        ],
        "DTMULT": [
            "DTMULT multiplier",
            "float",
            bs.sim.set_dtmult,
            "Sel multiplication factor for fast-time simulation",
        ],
        "ECHO": [
            "ECHO txt",
            "string",
            bs.scr.echo,
            "Show a text in command window for user to read",
        ],
        "FF": [
            "FF [timeinsec]",
            "[time]",
            bs.sim.fastforward,
            "Fast forward the simulation",
        ],
        "FILTERALT": [
            "FILTERALT ON/OFF,[bottom,top]",
            "bool,[alt,alt]",
            bs.scr.filteralt,
            "Display aircraft on only a selected range of altitudes",
        ],
        "FIXDT": [
            "FIXDT ON/OFF [tend]",
            "onoff,[time]",
            lambda flag, *args: bs.sim.ff(*args) if flag else bs.op(),
            "Legacy function for TMX compatibility",
        ],

        "GROUP": [
            "GROUP [grname, (areaname OR acid,...) ]",
            "[txt,acid/txt,...]",
            bs.traf.groups.group,
            "Add aircraft to a group. OR all aircraft in given area.\n"
            + "Returns list of groups when no argument is passed.\n"
            + "Returns list of aircraft in group when only a groupname is passed.\n"
            + "A group is created when a group with the given name doesn't exist yet.",
        ],
        "HOLD": ["HOLD", "", bs.sim.hold, "Pause(hold) simulation"],
        "IMPLEMENTATION": [
            "IMPLEMENTATION [base, implementation]",
            "[txt,txt]",
            select_implementation,
            "Select an alternate implementation for a Bluesky base class"
        ],
        "INSEDIT": [
            "INSEDIT txt",
            "string",
            bs.scr.cmdline,
            "Insert text op edit line in command window",
        ],
        "LEGEND": [
            "LEGEND label1, ..., labeln",
            "word,...",
            lambda *labels: plotter.legend(labels),
            "Add a legend to the last created plot",
        ],
        "LINE": [
            "LINE name,lat,lon,lat,lon",
            "txt,latlon,latlon",
            lambda name, *coords: areafilter.defineArea(name, "LINE", coords),
            "Draw a line on the radar screen",
        ],
        "LSVAR": [
            "LSVAR path.to.variable",
            "[word]",
            ve.lsvar,
            "Inspect any variable in a bluesky simulation",
        ],
        "MAGVAR": [
            "MAGVAR lat,lon",
            "lat,lon",
            bs.tools.geo.magdeccmd,
            "Show magnetic variation/declination at position",
        ],
        "MCRE": [
            "MCRE n, [type/*, alt/*, spd/*, dest/*]",
            "int,[txt,alt,spd,txt]",
            bs.traf.mcre,
            "Multiple random create of n aircraft in current view",
        ],
        "MOVE": [
            "MOVE acid,lat,lon,[alt,hdg,spd,vspd]",
            "acid,latlon,[alt,hdg,spd,vspd]",
            bs.traf.move,
            "Move an aircraft to a new position",
        ],
        "ND": [
            "ND acid",
            "txt",
            bs.scr.shownd,
            "Show navigation display with CDTI"
        ],
        "NOISE": [
            "NOISE [ON/OFF]",
            "[onoff]",
            bs.traf.setnoise,
            "Turbulence/noise switch",
        ],
        "OP": [
            "OP",
            "",
            bs.sim.op,
            "Start/Run simulation or continue after hold"
        ],
        "PAN": [
            "PAN latlon/acid/airport/waypoint/LEFT/RIGHT/ABOVE/DOWN",
            "pandir/latlon",
            bs.scr.pan,
            "Pan screen (move view) to a waypoint, direction or aircraft",
        ],
        "PLOT": [
            "PLOT [x], y [,dt,color,figure]",
            "[word,word,float,txt,int]",
            plotter.plot,
            "Create a graph of variables x versus y.",
        ],
        "POLY": [
            "POLY name,[lat,lon,lat,lon, ...]",
            "txt,[latlon,...]",
            lambda name, *coords: areafilter.defineArea(name, "POLY", coords),
            "Define a polygon-shaped area",
        ],
        "POLYALT": [
            "POLYALT name,top,bottom,lat,lon,lat,lon, ...",
            "txt,alt,alt,latlon,...",
            lambda name, top, bottom, *coords: areafilter.defineArea(
                name, "POLYALT", coords, top, bottom
            ),
            "Define a polygon-shaped area in 3D: between two altitudes",
        ],
        "POLYLINE": [
            "POLYLINE name,lat,lon,lat,lon,...",
            "txt,latlon,...",
            lambda name, *coords: areafilter.defineArea(name, "LINE", coords),
            "Draw a multi-segment line on the radar screen",
        ],
        "POS": [
            "POS acid/waypoint",
            "acid/wpt",
            bs.traf.poscommand,
            "Get info on aircraft, airport or waypoint",
        ],
        "QUIT": ["QUIT", "", bs.sim.stop, "Quit program/Stop simulation"],
        "REALTIME": [
            "REALTIME [ON/OFF]",
            "[bool]",
            bs.sim.realtime,
            "En-/disable realtime running allowing a variable timestep."],
        "RESET": ["RESET", "", bs.sim.reset, "Reset simulation"],
        "SEED": [
            "SEED value",
            "int",
            bs.sim.setseed,
            "Set seed for all functions using a randomizer (e.g.mcre,noise)",
        ],
        "SSD": [
            "SSD ALL/CONFLICTS/OFF or SSD acid0, acid1, ...",
            "txt,[...]",
            lambda *args: bs.scr.feature("SSD", args),
            "Show state-space diagram (=conflict prevention display/predictive ASAS)",
        ],
        "SWRAD": [
            "SWRAD GEO/GRID/APT/VOR/WPT/LABEL/ADSBCOVERAGE/TRAIL/POLY [dt]/[value]",
            "txt,[float]",
            bs.scr.feature,
            "Switch on/off elements and background of map/radar view",
        ],
        "SYMBOL": ["SYMBOL", "", bs.scr.symbol, "Toggle aircraft symbol"],
        "THR": [
            "THR acid, IDLE/0.0/throttlesetting/1.0/AUTO(default)",
            "acid[,txt]",
            bs.traf.setthrottle,
            "Set throttle or autotothrottle(default)",
        ],
        "TIME": [
            "TIME RUN(default) / HH:MM:SS.hh / REAL / UTC ",
            "[txt]",
            bs.sim.setutc,
            "Set simulated clock time",
        ],
        "TRAIL": [
            "TRAIL ON/OFF, [dt] OR TRAIL acid color",
            "[acid/bool],[float/txt]",
            bs.traf.trails.setTrails,
            "Toggle aircraft trails on/off",
        ],
        "UNGROUP": [
            "UNGROUP grname, acid",
            "txt,acid,...",
            bs.traf.groups.ungroup,
            "Remove aircraft from a group",
        ],

        "ZOOM": [
            "ZOOM IN/OUT or factor",
            "float/txt",
            lambda a: bs.scr.zoom(1.4142135623730951)
            if a == "IN"
            else bs.scr.zoom(0.7071067811865475)
            if a == "OUT"
            else bs.scr.zoom(a, True),
            "Zoom display in/out, you can also use +++ or -----",
        ],
    }

    #
    # Command synonym dictionary definea equivalent commands globally in stack
    #
    # Actual command definitions: see dictionary in def init(...) below
    #
    synonyms = {
        "ADDAIRWAY": "ADDAWY",
        "AWY": "POS",
        "AIRPORT": "POS",
        "AIRWAYS": "AIRWAY",
        "BANKLIM": "BANK",
        "CHDIR": "CD",
        "COL": "COLOR",
        "COLOUR": "COLOR",
        "CONTINUE": "OP",
        "CREATE": "CRE",
        "CLOSE": "QUIT",
        "DEBUG": "CALC",
        "DELETE": "DEL",
        "DISP": "SWRAD",
        "END": "QUIT",
        "EXIT": "QUIT",
        "FWD": "FF",
        "IMPL": "IMPLEMENTATION",
        "IMPLEMENT": "IMPLEMENTATION",
        "LINES": "POLYLINE",
        "MAGDEC": "MAGVAR",
        "MAGDECL":"MAGVAR",
        "PAUSE": "HOLD",
        "POLYGON": "POLY",
        "POLYLINES": "POLYLINE",
        "PRINT": "ECHO",
        "Q": "QUIT",
        "RT": "REALTIME",
        "RTF": "DTMULT",
        "STOP": "QUIT",
        "RUN": "OP",
        "RUNWAYS": "POS",
        "SAVE": "SAVEIC",
        "START": "OP",
        "TRAILS": "TRAIL",
        "VAR": "MAGVAR"
    }

    append_commands(cmddict, synonyms)


def singbluesky():
    webbrowser.open_new("https://youtu.be/aQUlA8Hcv4s")
    return True


def distcalc(lat0, lon0, lat1, lon1):
    try:
        qdr, dist = geo.qdrdist(lat0, lon0, lat1, lon1)
        return True, "QDR = %.2f deg, Dist = %.3f nm" % (qdr % 360.0, dist)
    except:
        return False, "Error in dist calculation."


def setscenpath(newpath):

    if not newpath:
        return False, "Needs an absolute or relative path"

    newpath = Path(newpath)
    # If this is a relative path we need to prefix scenario folder
    if not newpath.is_absolute():
        newpath = bs.resource(settings.scenario_path) / newpath

    if not newpath.is_dir():
        return False, "Error: cannot find path: " + newpath

    # Change path
    settings.scenario_path = newpath

    return True
