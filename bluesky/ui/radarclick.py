from ..tools.misc import findnearest, cmdsplit
from math import cos, atan2, radians, degrees


def radarclick(cmdline, lat, lon, traf, navdb):
    """Process lat,lon as clicked in radar window"""
    tostack = ''
    todisplay = cmdline

    # Specify which argument can be clicked, and how, in this dictionary
    # and when it's the last, also add ENTER

    clickcmd = {"": "acid,-",
                "POS": "acid",
                "CRE":  "-,-,latlon,-,hdg,-,-",
                "HDG": "acid,hdg",
                "SPD": "acid,-",
                "ALT": "acid,-",
                "LISTRTE": "acid,-",
                "ADDWPT": "acid,latlon,-,-,-",
                "ASAS": "acid,-",
                "DEl": "acid,-",
                "LNAV": "acid,-",
                "VNAV": "acid,-",
                "VS": "acid,-",
                "ND": "acid",
                "NAVDISP": "acid",
                "ASAS": "acid,-",
                "ORIG": "acid,apt",
                "DEST": "acid,apt",
                "PAN": "latlon",
                "MOVE": "acid,latlon,-,-,hdg",
                "DIST": "latlon,-,latlon",
                "LINE": "latlon,-,latlon",
                "AREA": "latlon,-,latlon",
                }

    cmdargs = cmdsplit(cmdline)
    numargs = len(cmdargs) - 1

    # Save command
    if numargs >= 0:
        cmd = cmdargs[0]
    else:
        # avoid negative nr of args when there is no cmd
        cmd = ""
        numargs = 0

    # Check for syntax of acid first in command line:
    # (as "HDG acid,hdg"  and "acid HDG hdg" are both a correct syntax
    if numargs >= 1:
        if cmd != "" and traf.id.count(cmd) > 0:
            acid = cmd
            cmd = cmdargs[1]
            cmdargs[1] = acid

        if numargs >= 1:
            acid = cmdargs[1]

    # -------- Process click --------
    # Double click on aircraft = POS command
    if cmd != "" and numargs == 0 and traf.id.count(cmdargs[0]) > 0:
        todisplay = '\n'                 # Clear the current command
        tostack = "POS " + cmdargs[0]  # And send a pos command to the stack

    # Insert: nearestaircraft id
    else:

        # Find command in clickcmd dictionary
        try:
            lookup = clickcmd[cmd]
        except KeyError:
            lookup = False

        if lookup:

            # Detrmine argument click type
            clickargs = lookup.lower().split(",")
            if numargs < len(clickargs):
                clicktype = clickargs[numargs]

                if clicktype == "acid":
                    idx = findnearest(lat, lon, traf.lat, traf.lon)
                    if idx >= 0:
                        todisplay = traf.id[idx] + " "

                elif clicktype == "latlon":
                    todisplay = " " + str(round(lat, 6)) + "," + str(round(lon, 6)) + " "

                elif clicktype == "apt":
                    idx = findnearest(lat, lon, navdb.aplat, navdb.aplon)
                    if idx >= 0:
                        todisplay = navdb.apid[idx] + " "

                elif clicktype == "hdg":
                    # Read start position from command line
                    if cmd == "CRE":
                        try:
                            reflat = float(cmdargs[3])
                            reflon = float(cmdargs[4])
                            synerr = False
                        except:
                            synerr = True
                    elif cmd == "MOVE":
                        try:
                            reflat = float(cmdargs[2])
                            reflon = float(cmdargs[3])
                            synerr = False
                        except:
                            synerr = True
                    else:
                        if traf.id.count(acid) > 0:
                            idx = traf.id.index(acid)
                            reflat = traf.lat[idx]
                            reflon = traf.lon[idx]
                            synerr = False
                        else:
                            synerr = True
                    if not synerr:
                        dy = lat - reflat
                        dx = (lon - reflon) * cos(radians(reflat))
                        hdg = degrees(atan2(dx, dy)) % 360.

                        todisplay = " " + str(int(hdg)) + " "

                # Is it the last argument? (then we will insert ENTER as well)
                if numargs + 1 >= len(clickargs):
                    tostack = cmdline + todisplay
                    todisplay = todisplay + '\n'

        # No clickable command (cmd not in dictionary)
        else:
            tostack   = ""
            todisplay = ""
            
    return tostack, todisplay
