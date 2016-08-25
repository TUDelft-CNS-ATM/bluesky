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
import sys

from ..tools import geo
from ..tools.aero import kts, ft, fpm, tas2cas, density
from ..tools.misc import txt2alt, cmdsplit, txt2lat, txt2lon
from .. import settings

# Global variables
cmddict   = dict()
cmdsynon  = dict()
cmdstack  = []

scenfile  = ""
scentime  = []
scencmd   = []


# ------------------ [start] Deprecated -------------------
# An alternative way to add your own commands:
# add your entry to the dictionary.
# The dictionary should be formed as {"Key":module'}.

# "Key" is a FOUR-symbol reference used at the start of command.
# 'module' is the name of the .py-file in which the
# commands are located (without .py).

# Make sure that the module has a function "process" with
# arguments:
#   command, number of args, array of args, sim, traf, scr, cmd

extracmdmodules = {
    "SYN_": 'synthetic'
}

# Import modules from the list
extracmdrefs = {}
sys.path.append('bluesky/stack/')
for key in extracmdmodules:
    obj=__import__(extracmdmodules[key],globals(),locals(),[],0)
    extracmdrefs[key]=obj
# ------------------ [end] Deprecated -------------------


def init(sim, traf, scr):
    """ Initialization of the default stack commands. This function is called
        at the initialization of the main simulation object."""

    #Command dictionary: command, helptext, arglist, function to call
    # Enclose optional arguments with []
    # Separate argument type variants with /
    #--------------------------------------------------------------------
    commands = {
        "ADDNODES": [
            "ADDNODES number",
            "int",
            sim.addNodes],
        "ADDWPT": [
            "ADDWPT acid, (wpname/lat,lon),[alt],[spd],[afterwp]",
            "acid,pos/txt,[alt,spd,txt]",
            # lambda: short-hand for using function output as argument, equivalent with:
            #
            # def fun(idx, args):
            #     return traf.route[idx].addwptStack(traf, idx, *args)
            # fun(idx,*args)
            lambda idx, *args: traf.route[idx].addwptStack(traf, idx, *args)
        ],
        "ALT": [
            "ALT acid, alt, [vspd]",
            "acid,alt,[vspd]",
            traf.selalt
        ],
        "AREA": [
            "AREA OFF, or\nlat0,lon0,lat1,lon1[,lowalt]\nor\nAREA FIR,radius[,lowalt]\nor\nAREA CIRCLE,lat0,lon0,radius[,lowalt]",
            "float/txt,float,[float,float,float]",
            lambda *args: traf.setArea(scr, sim.metric, *args)
        ],
        "ASAS": [
            "ASAS ON/OFF",
            "[onoff]",
            traf.asas.toggle
        ],
        "BATCH": [
            "BATCH filename",
            "string",
            sim.batch],
        "BENCHMARK": [
            "BENCHMARK [scenfile,time]",
            "[txt,time]",
            sim.benchmark
        ],
        "BOX": [
            "BOX name,lat,lon,lat,lon",
            "txt,latlon,latlon",
            lambda name, *coords: scr.objappend(2, name, coords)
        ],
        "CALC": [
            "CALC expression",
            "string",
            lambda expr: scr.echo("Ans = " + str(eval(expr)))
        ],
        "CDMETHOD": [
            "CDMETHOD [method]",
            "[txt]",
            traf.asas.SetCDmethod
        ],
        "CIRCLE": [
            "CIRCLE name,lat,lon,radius",
            "txt,latlon,float",
            lambda name, *coords: scr.objappend(3, name, coords)
        ],
        "CRE": [
            "CRE acid,type,lat,lon,hdg,alt,spd",
            "txt,txt,pos,hdg,alt,spd",
            traf.create
        ],
        "DEL": [
            "DEL acid/WIND/shape",
            "txt",
            lambda a:   traf.delete(a)    if traf.id.count(a) > 0 \
                   else traf.wind.clear() if a=="WIND" \
                   else scr.objappend(0, a, None)
        ],
        "DELWPT": [
            "DELWPT acid,wpname",
            "acid,txt",
            lambda idx, wpname: traf.route[idx].delwpt(wpname)
        ],
        "DEST": [
            "DEST acid, latlon/airport",
            "acid,pos",
            lambda idx, *args: traf.setDestOrig("DEST", idx, *args)
        ],
        "DIRECT": [
            "DIRECT acid wpname",
            "acid,txt",
            lambda idx, wpname: traf.route[idx].direct(traf, idx, wpname)
        ],
        "DIST": [
            "DIST lat0, lon0, lat1, lon1",
            "latlon,latlon",
            lambda *args: scr.echo("QDR = %.2f deg, Dist = %.3f nm" % geo.qdrdist(*args))
        ],
        "DT": [
            "DT dt",
            "float",
            sim.setDt
        ],
        "DTLOOK": [
            "DTLOOK [time]",
            "[float]",
            traf.asas.SetDtLook
        ],
        "DTMULT": [
            "DTMULT multiplier",
            "float",
            sim.setDtMultiplier
        ],
        "DTNOLOOK": [
            "DTNOLOOK [time]",
            "[float]",
            traf.asas.SetDtNoLook
        ],
        "DUMPRTE": [
            "DUMPRTE acid",
            "acid",
            lambda idx: traf.route[idx].dumpRoute(traf, idx)
        ],
        "ECHO": [
            "ECHO txt",
            "string",
            scr.echo
        ],
        "ENG": [
            "ENG acid,[engine_id]",
            "acid,[txt]",
            traf.perf.engchange
        ],
        "FF": [
            "FF [tend]",
            "[time]",
            sim.fastforward
        ],
        "FIXDT": [
            "FIXDT ON/OFF [tend]",
            "onoff,[time]",
            sim.setFixdt
        ],
        "GETWIND": [
            "GETWIND lat,lon[,alt]",
            "latlon,[alt]",
            traf.wind.get
        ],
        "HDG": [
            "HDG acid,hdg (deg,True)",
            "acid,float",
            traf.selhdg
        ],
        "HELP": [
            "HELP [command]",
            "[txt]",
            lambda *args: scr.echo(helptext(*args))
        ],
        "HOLD": [
            "HOLD",
            "",
            sim.pause
        ],
        "IC": [
            "IC [IC/filename]",
            "[string]",
            lambda *args: ic(scr, sim, *args)
        ],
        "INSEDIT": [
            "INSEDIT txt",
            "string",
            scr.cmdline
        ],
        "LINE": [
            "LINE name,lat,lon,lat,lon",
            "txt,latlon,latlon",
            lambda name, *coords: scr.objappend(1, name, coords)
        ],
        "LISTRTE": [
            "LISTRTE acid, [pagenr]",
            "acid,[int]",
            lambda idx, *args: traf.route[idx].listrte(scr, idx, traf, *args)
        ],
        "LNAV": [
            "LNAV acid,[ON/OFF]",
            "acid,[onoff]",
            traf.setLNAV
        ],
        "MCRE": [
            "MCRE n, [type/*, alt/*, spd/*, dest/*]",
            "int,[txt,alt,spd,txt]",
            lambda *args: traf.mcreate(*args, area=scr.getviewlatlon())
        ],
        "METRIC": [
            "METRIC OFF/0/1/2, [dt]",
            "onoff/int,[float]",
            lambda *args: sim.metric.toggle(traf, *args)
        ],
        "MOVE": [
            "MOVE acid,lat,lon,[alt,hdg,spd,vspd]",
            "acid,latlon,[alt,hdg,spd,vspd]",
            traf.move
        ],
        "ND": [
            "ND acid",
            "txt",
            lambda acid: scr.feature("ND", acid)
        ],
        "NOISE": [
            "NOISE [ON/OFF]",
            "[onoff]",
            traf.setNoise
        ],
        "NOM": [
            "NOM acid",
            "acid",
            traf.nom
        ],
        "NORESO": [
            "NORESO [acid]",
            "[string]",
            traf.asas.SetNoreso
        ],
        "OP": [
            "OP",
            "",
            sim.start
        ],
        "ORIG": [
            "ORIG acid, latlon/airport",
            "acid,pos",
            lambda *args: traf.setDestOrig("ORIG", *args)
        ],
        "PAN": [
            "PAN latlon/acid/airport/waypoint",
            "pos/txt",
            scr.pan
        ],
        "PCALL": [
            "PCALL filename [REL/ABS]",
            "txt,[txt]",
            lambda *args: openfile(*args, mergeWithExisting=True)
        ],

        "POLY": [
            "POLY name,lat,lon,lat,lon, ...",
            "txt,latlon,...",
            lambda name, *coords: scr.objappend(4, name, coords)
        ],
        "POS": [
            "POS acid",
            "txt",
            lambda acid: scr.showacinfo(acid, traf.acinfo(acid))
        ],
        "PRIORULES": [
            "PRIORULES [ON/OFF PRIOCODE]",
            "[onoff, txt]",
            traf.asas.SetPrio
        ],
        "RESET": [
            "RESET",
            "",
            sim.reset
        ],
        "RFACH": [
            "RFACH [factor]",
            "[float]",
            traf.asas.SetResoFacH
        ],
        "RFACV": [
            "RFACV [factor]",
            "[float]",
            traf.asas.SetResoFacV
        ],
        "RESO": [
            "RESO [method]",
            "[txt]",
            traf.asas.SetCRmethod
        ],
        "RESOOFF": [
            "RESOOFF [acid]",
            "[string]",
            traf.asas.SetResooff
        ],
        "RMETHH": [
            "RMETHH [method]",
            "[txt]",
            traf.asas.SetResoHoriz
        ],
        "RMETHV": [
            "RMETHV [method]",
            "[txt]",
            traf.asas.SetResoVert
        ],
        "RSZONEDH": [
            "RSZONEDH [height]",
            "[float]",
            traf.asas.SetPZHm
        ],
        "RSZONER": [
            "RSZONER [radius]",
            "[float]",
            traf.asas.SetPZRm
        ],
        "RUNWAYS": [
            "RUNWAYS ICAO",
            "txt",
            lambda ICAO: traf.navdb.listrwys(ICAO)
        ],
        "SAVEIC": [
            "SAVEIC filename",
            "string",
            lambda fname: saveic(fname, sim, traf)
        ],
        "SCEN": [
            "SCEN scenname",
            "string",
            sim.scenarioInit
        ],
        "SEED": [
            "SEED value",
            "int",
            setSeed],
        "SPD": [
            "SPD acid,spd (CAS-kts/Mach)",
            "acid,spd",
            traf.selspd
        ],
        "SSD": [
            "SSD acid/ALL/OFF",
            "txt",
            scr.showssd
        ],
        "STOP": [
            "STOP",
            "",
            sim.stop
        ],
        "SWRAD": [
            "SWRAD GEO/GRID/APT/VOR/WPT/LABEL/ADSBCOVERAGE/TRAIL [dt]/[value]",
            "txt,[float]",
            scr.feature
        ],
        "SYMBOL": [
            "SYMBOL",
            "",
            scr.symbol
        ],
        "TAXI": [
            "TAXI ON/OFF : OFF auto deletes traffic below 1500 ft",
            "onoff",
            traf.setTaxi
        ],
        "TRAIL": [
            "TRAIL ON/OFF, [dt] OR TRAIL acid color",
            "acid/bool,[float/txt]",
            traf.setTrails
        ],
        "VNAV": [
            "VNAV acid,[ON/OFF]",
            "acid,[onoff]",
            traf.setVNAV
        ],
        "VS": [
            "VS acid,vspd (ft/min)",
            "acid,vspd",
            traf.selvspd
        ],
        "WIND": [
            "WIND lat,lon,alt/*,dir,spd[,alt,dir,spd,alt,...]",
            "latlon,[alt],float,float,...,...,...",   # last 3 args are repeated
            traf.wind.add
        ],
        "ZONEDH": [
            "ZONEDH [height]",
            "[float]",
            traf.asas.SetPZH
        ],
        "ZONER": [
            "ZONER [radius]",
            "[float]",
            traf.asas.SetPZR
        ],
        "ZOOM": [
            "ZOOM IN/OUT or factor",
            "float/txt",
            lambda a: scr.zoom(1.4142135623730951) if a == "IN" else \
                      scr.zoom(0.7071067811865475) if a == "OUT" else \
                      scr.zoom(a, True)
        ]
    }

    cmddict.update(commands)

    #--------------------------------------------------------------------
    # Command synonym dictionary
    synonyms = {
        "CONTINUE": "OP",
        "CREATE": "CRE",
        "DELETE": "DEL",
        "DIRECTTO": "DIRECT",
        "DIRTO": "DIRECT",
        "DISP": "SWRAD",
        "END": "STOP",
        "EXIT": "STOP",
        "FWD": "FF",
        "HMETH": "RMETHH",
        "HRESOM": "RMETHH",
        "HRESOMETH": "RMETHH",
        "PAUSE": "HOLD",
        "Q": "STOP",
        "QUIT": "STOP",
        "RUN": "OP",
        "RESOFACH": "RFACH",
        "RESOFACV": "RFACV",
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


def get_scenfile():
    return scenfile


def get_scendata():
    return scentime, scencmd


def set_scendata(newtime, newcmd):
    global scentime, scencmd
    scentime = newtime
    scencmd  = newcmd


def append_commands(newcommands):
    """ Append additional functions to the stack command dictionary """
    cmddict.update(newcommands)


def helptext(cmd=''):
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
    elif cmd in cmddict:
        return cmddict[cmd][0]
    else:
        return "HELP: Unknown command: " + cmd


def setSeed(value):
    seed(value)
    np.random.seed(value)


def reset():
    global scentime, scencmd
    scentime = []
    scencmd  = []


def stack(cmdline):
    # Stack one or more commands separated by ";"
    cmdline = cmdline.strip()
    if len(cmdline) > 0:
        for line in cmdline.split(';'):
            cmdstack.append(line)


def openfile(scenname, absrel='ABS', mergeWithExisting=False):
    global scentime, scencmd

    # If timestamps in file should be interpreted as relative we need to add
    # the current simtime to every timestamp
    t_offset = sim.simt if absrel == 'REL' else 0.0

    # Add .scn extension if necessary
    if scenname.lower().find(".scn") < 0:
        scenname = scenname + ".scn"

    # If it is with a path don't touch it, else add path
    if scenname.find("/") < 0 and scenname.find( "\\") < 0:
        scenfile = settings.scenario_path
        if scenfile[-1] is not '/':
            scenfile += '/'
        scenfile += scenname
    else:
        scenfile = scenname

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
                    tstamp = line[:icmdline]
                    ttxt = tstamp.strip().split(':')
                    ihr = int(ttxt[0]) * 3600.0
                    imin = int(ttxt[1]) * 60.0
                    xsec = float(ttxt[2])
                    scentime.append(ihr + imin + xsec + t_offset)
                    scencmd.append(line[icmdline + 1:-1])
                except:
                    if not(len(line.strip())>0 and line.strip()[0]=="#"):                        
                        print "except this:", line
                    pass  # nice try, we will just ignore this syntax error

    if mergeWithExisting:
        # If we are merging we need to sort the resulting command list
        scentime, scencmd = [list(x) for x in zip(*sorted(
            zip(scentime, scencmd), key=lambda pair: pair[0]))]

    return True


def ic(scr, sim, filename=''):
    global scenfile
    if filename == '':
        filename = scr.show_file_dialog()
    elif filename == "IC":
        filename = scenfile

    if len(filename) > 0:
        sim.reset()
        result = openfile(filename)
        if type(result) is bool:
            scenfile = filename
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
                  repr(traf.hdg[i]) + "," + repr(traf.alt[i] / ft) + "," + \
                  repr(tas2cas(traf.tas[i], traf.alt[i]) / kts)

        f.write(timtxt + cmdline + chr(13) + chr(10))

        # VS acid,vs
        if abs(traf.vs[i]) > 0.05:  # 10 fpm dead band
            if abs(traf.avs[i]) > 0.05:
                vs_ = traf.avs[i] / fpm
            else:
                vs_ = traf.vs[i] / fpm

            cmdline = "VS " + traf.id[i] + "," + repr(vs_)
            f.write(timtxt + cmdline + chr(13) + chr(10))

        # Autopilot commands
        # Altitude
        if abs(traf.alt[i] - traf.aalt[i]) > 10.:
            cmdline = "ALT " + traf.id[i] + "," + repr(traf.aalt[i] / ft)
            f.write(timtxt + cmdline + chr(13) + chr(10))

        # Heading as well when heading select
        delhdg = (traf.hdg[i] - traf.ahdg[i] + 180.) % 360. - 180.
        if abs(delhdg) > 0.5:
            cmdline = "HDG " + traf.id[i] + "," + repr(traf.ahdg[i])
            f.write(timtxt + cmdline + chr(13) + chr(10))

        # Speed select? => Record
        rho = density(traf.alt[i])  # alt in m!
        aptas = sqrt(1.225 / rho) * traf.aspd[i]
        delspd = aptas - traf.tas[i]

        if abs(delspd) > 0.4:
            cmdline = "SPD " + traf.id[i] + "," + repr(traf.aspd[i] / kts)
            f.write(timtxt + cmdline + chr(13) + chr(10))

        # DEST acid,dest-apt
        if traf.dest[i] != "":
            cmdline = "DEST " + traf.id[i] + "," + traf.dest[i]
            f.write(timtxt + cmdline + chr(13) + chr(10))

        # ORIG acid,orig-apt
        if traf.orig[i] != "":
            cmdline = "ORIG " + traf.id[i] + "," + \
                      traf.orig[i]
            f.write(timtxt + cmdline + chr(13) + chr(10))

    # Saveic: should close
    f.close()
    return True


def process(sim, traf, scr):
    """process and empty command stack"""
    global cmdstack

    # Process stack of commands
    for line in cmdstack:
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
        if cmd in cmdsynon.keys():
            cmd = cmdsynon[cmd]

        if cmd in cmddict.keys():
            helptext, argtypelist, function = cmddict[cmd]
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
                arglist = [line[len(cmd) + 1:]]
            else:
                arglist = []
                curtype = curarg = 0
                while curtype < len(argtypes) and curarg < len(args):
                    if argtypes[curtype][:3] == '...':
                        repeatsize = len(argtypes) - curtype
                        curtype = curtype - repeatsize
                    argtype    = argtypes[curtype].strip().split('/')
                    for i in range(len(argtype)):
                        try:
                            argtypei = argtype[i]
                            parsed_arg, opt_arg, argstep = argparse(argtypei, curarg, args, traf, scr)
                            if parsed_arg[0] is None and argtypei in optargs:
                                arglist += optargs[argtypei]
                            else:
                                arglist += parsed_arg
                            optargs.update(opt_arg)
                            curarg  += argstep
                            break
                        except:
                            # not yet last type possible here?
                            if i < len(argtype) - 1:
                                # We have alternative argument formats that we can try
                                continue
                            else:
                                synerr = True
                                scr.echo("Syntax error in processing arguments")
                                scr.echo(line)
                                scr.echo(helptext)
                    curtype += 1

            # Call function return flag,text
            # flag: indicates sucess
            # text: optional error message
            if not synerr:
                results = function(*arglist)  # * = unpack list to call arguments

                if type(results) == bool:  # Only flag is returned
                    synerr = not results
                    if synerr:
                        if numargs <= 0 or args[curarg] == "?":
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
        elif cmd[:4] in extracmdrefs:
            extracmdrefs[cmd[:4]].process(cmd[4:], numargs, [cmd] + args, sim, traf, scr, self)

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
    """ Parse one or more arguments.

        Returns:
        - A list with the parse results
        - The number of arguments parsed
        - A dict with additional optional parsed arguments. """
    if args[argidx] == "" or args[argidx] == "*":  # Empty arg or wildcard => parse None
        return [None], {}, 1

    if argtype == "acid":  # aircraft id => parse index
        idx = traf.id2idx(args[argidx])
        if idx < 0:
            scr.echo(cmd + ":" + args[idx] + " not found")
            raise IndexError
        else:
            return [idx], {}, 1

    if argtype == "txt":  # simple text
        return [args[argidx]], {}, 1

    if argtype == "float":  # float number
        return [float(args[argidx])], {}, 1

    if argtype == "int":   # integer
        return [int(args[argidx])], {}, 1

    if argtype == "onoff" or argtype == "bool":
        sw = (args[argidx] == "ON" or
              args[argidx] == "1" or args[argidx] == "TRUE")
        return [sw], {}, 1

    if argtype == "pos":
        # Arg is an existing aircraft?
        idx = traf.id2idx(args[argidx])
        if idx >= 0:
            return [traf.lat[idx], traf.lon[idx]], {}, 1
        # Arg is an airport?
        idx = traf.navdb.getapidx(args[argidx])
        if idx >= 0:
            # Next arg is a runway?
            if len(args) > argidx + 1 and args[argidx] in traf.navdb.rwythresholds and \
                    args[argidx + 1] in traf.navdb.rwythresholds[args[argidx]]:
                arglist = traf.navdb.rwythresholds[args[argidx]][args[argidx + 1]][:2]
                optargs = {"hdg": [traf.navdb.rwythresholds[args[argidx]][args[argidx + 1]][2]]}
                return arglist, optargs, 2

            # If no runway return airport center
            return [traf.navdb.aplat[idx], traf.navdb.aplon[idx]], {}, 1
        # Arg is a waypoint?
        idx = traf.navdb.getwpidx(args[argidx])
        if idx >= 0:
            return [traf.navdb.wplat[idx], traf.navdb.wplon[idx]], {}, 1
        # Arg, next arg are a lat/lon combination
        return [txt2lat(args[argidx]), txt2lon(args[argidx + 1])], {}, 2

    if argtype == "latlon":
        return [txt2lat(args[argidx]), txt2lon(args[argidx + 1])], {}, 2

    if argtype == "spd":  # CAS[kts] Mach
        spd = float(args[argidx].upper().replace("M", ".").replace("..", "."))
        if not 0.1 < spd < 1.0:
            spd *= kts
        return [spd], {}, 1  # speed CAS[m/s] or Mach (float)

    if argtype == "vspd":
        return [fpm * float(args[argidx])], {}, 1

    if argtype == "alt":  # alt: FL250 or 25000 [ft]
        return [ft * txt2alt(args[argidx])], {}, 1  # alt in m

    if argtype == "hdg":
        # TODO: for now no difference between magnetic/true heading
        hdg = float(args[argidx].upper().replace('T', '').replace('M', ''))
        return [hdg], {}, 1

    if argtype == "time":
        ttxt = args[argidx].strip().split(':')
        if len(ttxt) >= 3:
            ihr  = int(ttxt[0]) * 3600.0
            imin = int(ttxt[1]) * 60.0
            xsec = float(ttxt[2])
            return [ihr + imin + xsec], {}, 1
        else:
            return [float(args[argidx])], {}, 1
