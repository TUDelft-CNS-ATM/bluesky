from findnearest import findnearest
from math import cos, atan2, radians, degrees


def radarclick(cmdline, lat, lon, traf):
    """Process click in radar window"""
    tostack   = ''
    todisplay = cmdline

    # Specify which argument can be clicked, and how, in this dictionary
    # and when it's the last, also add ENTER

    clickcmd = {"POS": "acid",
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

    while cmdline.find(",,") >= 0:
        cmdline = cmdline.replace(",,", ",@,")  # Mark empty arguments

    # Replace comma's by space
    cmdline = cmdline.replace(",", " ")

    # Split using spaces
    cmdargs = cmdline.split()     # Make list of cmd arguments

    # Adjust for empty arguments
    for i in range(len(cmdargs)):
        if cmdargs[i] == "@":
            cmdargs[i] = ""
    numargs = len(cmdargs) - 1

    # Save command
    if numargs >= 0:
        cmd = cmdargs[0]
    else:
        cmd = ""

    # Check for acid first in command line:
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
    if numargs == 0 and traf.id.count(cmdargs[0]) > 0:
        todisplay = '\n'                 # Clear the current command
        tostack   = "POS " + cmdargs[0]  # And send a pos command to the stack

    # No command: insert nearest aircraft id
    elif cmd == "":
        idx = findnearest(traf, lat, lon)
        if idx >= 0:
            todisplay = traf.id[idx] + " "

    # Insert: nearestaircraft id
    else:

        # Find command in clickcmd dictionary
        lookup = clickcmd[cmd]
        if lookup:

            # Detrmine argument click type
            clickargs = lookup.lower().split(",")
            if numargs < len(clickargs):
                clicktype = clickargs[numargs]

                if clicktype == "acid":
                    idx = findnearest(traf, lat, lon)
                    if idx >= 0:
                        todisplay = traf.id[idx] + " "

                elif clicktype == "latlon":
                    todisplay = " " + str(round(lat, 6)) + "," + str(round(lon, 6)) + " "

                elif clicktype == "hdg":
                    # Read start position from command line
                    if cmd == "CRE":
                        try:
                            clat = float(cmdargs[3])
                            clon = float(cmdargs[4])
                            synerr = False
                        except:
                            synerr = True
                    elif cmd == "MOVE":
                        try:
                            clat = float(cmdargs[2])
                            clon = float(cmdargs[3])
                            synerr = False
                        except:
                            synerr = True
                    else:
                        if traf.id.count(acid) > 0:
                            idx = traf.id.index(acid)
                            clat = traf.lat[idx]
                            clon = traf.lon[idx]
                            synerr = False
                        else:
                            synerr = True
                    if not synerr:
                        dy = lat - clat
                        dx = (lon - clon) * cos(radians(clat))
                        hdg = degrees(atan2(dx, dy)) % 360.

                        todisplay = " " + str(int(hdg)) + " "

                # Is it the last argument? (then we will insert ENTER as well)
                if numargs + 1 >= len(clickargs):
                    tostack = cmdline + todisplay
                    todisplay = todisplay + '\n'

    return tostack, todisplay
