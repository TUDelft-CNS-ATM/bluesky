''' BlueSky scenario recorder. '''
import math
from pathlib import Path

import bluesky as bs
from bluesky.tools.aero import kts, ft, fpm, tas2cas, density
from bluesky.tools.misc import tim2txt
from bluesky.stack.cmdparser import command, commandgroup

# When SAVEIC is used, we will also have a recording scenario file handle
savefile = None  # File object of recording scenario file
defexcl = [
    "PAN",
    "ZOOM",
    "HOLD",
    "POS",
    "INSEDIT",
    "SAVEIC",
    "QUIT",
    "PCALL",
    "CALL",
    "PLOT",
    "CALC",
    "FF",
    "IC",
    "OP",
    "HOLD",
    "RESET",
    "MCRE",
    "CRE",
    "TRAFGEN",
    "LISTRTE",
]  # Commands to be excluded, default
# Note (P)CALL is always excluded! Commands in called file are saved explicitly
saveexcl = defexcl
# simt time of moment of SAVEIC command, 00:00:00.00 in recorded file
saveict0 = 0.0

@commandgroup
def saveic(filename: 'word' = ''):
    """ Save the current traffic realization in a scenario file. """
    global savefile, saveict0

    # No args? Give current status
    if not filename:
        if savefile is None:
            return False, "SAVEIC is not running"
        else:
            return True, "SAVEIC is already on\n" + "File: " + savefile.name

    # If recording is already on, give message
    if savefile is not None:
        return False, "SAVEIC is already on\n" + "Savefile:  " + savefile.name

    # Add extension .scn if not already present
    filename = Path(filename).with_suffix('.scn')
    # If it is with path don't touch it, else add path
    if not filename.is_absolute():
        filename = bs.resource(bs.settings.scenario_path) / filename

    try:
        f = open(filename, "w")
    except:
        return False, "Error writing to file"

    # Write files
    timtxt = "00:00:00.00>"  # Current time will be zero
    saveict0 = bs.sim.simt

    for i in range(bs.traf.ntraf):
        # CRE acid,type,lat,lon,hdg,alt,spd
        cmdline = (f'CRE {bs.traf.id[i]},{bs.traf.type[i]},{bs.traf.lat[i]},'
                   f'{bs.traf.lon[i]},{bs.traf.trk[i]},{bs.traf.alt[i] / ft},'
                   f'{tas2cas(bs.traf.tas[i], bs.traf.alt[i]) / kts}')

        f.write(timtxt + cmdline + "\n")

        # VS acid,vs
        if abs(bs.traf.vs[i]) > 0.05:  # 10 fpm dead band
            if abs(bs.traf.ap.vs[i]) > 0.05:
                vs_ = bs.traf.ap.vs[i] / fpm
            else:
                vs_ = bs.traf.vs[i] / fpm

            cmdline = f'VS {bs.traf.id[i]},{vs_}'
            f.write(timtxt + cmdline + "\n")

        # Autopilot commands
        # Altitude
        if abs(bs.traf.alt[i] - bs.traf.ap.alt[i]) > 10.0:
            cmdline = "ALT " + bs.traf.id[i] + \
                "," + repr(bs.traf.ap.alt[i] / ft)
            f.write(timtxt + cmdline + "\n")

        # Heading as well when heading select
        delhdg = (bs.traf.hdg[i] - bs.traf.ap.trk[i] + 180.0) % 360.0 - 180.0
        if abs(delhdg) > 0.5:
            cmdline = "HDG " + bs.traf.id[i] + "," + repr(bs.traf.ap.trk[i])
            f.write(timtxt + cmdline + "\n")

        # Speed select? => Record
        rho = density(bs.traf.alt[i])  # alt in m!
        aptas = math.sqrt(1.225 / rho) * bs.traf.ap.spd[i]
        delspd = aptas - bs.traf.tas[i]

        if abs(delspd) > 0.4:
            cmdline = "SPD " + bs.traf.id[i] + \
                "," + repr(bs.traf.ap.spd[i] / kts)
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

            # add other waypoints
            cmdline = "ADDWPT " + bs.traf.id[i] + " "
            wpname = route.wpname[iwp]
            if wpname[: len(bs.traf.id[i])] == bs.traf.id[i]:
                wpname = repr(route.wplat[iwp]) + "," + repr(route.wplon[iwp])
            cmdline = cmdline + wpname + ","

            if route.wpalt[iwp] >= 0.0:
                cmdline = cmdline + repr(route.wpalt[iwp] / ft) + ","
            else:
                cmdline = cmdline + ","

            if route.wpspd[iwp] >= 0.0:
                if route.wpspd[iwp] > 1.0:
                    cmdline = cmdline + repr(route.wpspd[iwp] / kts)
                else:
                    cmdline = cmdline + repr(route.wpspd[iwp])

            f.write(timtxt + cmdline + "\n")

    # Saveic: save file
    savefile = f
    return True


@saveic.subcommand(name='EXCEPT')
def setexcept(*commands: 'txt'):
    ''' Indicate commands that need to be omitted by SAVEIC. '''
    global saveexcl
    if not commands:
        return True, "EXCEPT is now: " + " ".join(saveexcl)

    if 'NONE' in commands:
        # All commands should be saved, only skip the bare minimum
        saveexcl = ["INSEDIT", "SAVEIC"]
    else:
        saveexcl = command
    return True


@saveic.subcommand(name='CLOSE')
def saveclose():
    ''' Reset recorder. '''
    global savefile
    if savefile is not None:
        savefile.close()

    savefile = None
    return True


def savecmd(cmd, line):
    ''' Save command line to file if SAVEIC is turned on. '''
    if savefile is None or cmd in saveexcl:
        return

    # Write in format "HH:MM:SS.hh>
    timtxt = tim2txt(bs.sim.simt - saveict0)
    savefile.write(f'{timtxt}>{line}\n')


def reset():
    ''' Reset SAVEIC recorder: close file and reset excluded command list. '''
    global saveexcl
    saveclose()
    saveexcl = defexcl