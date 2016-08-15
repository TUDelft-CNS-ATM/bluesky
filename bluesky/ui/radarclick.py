from ..tools import geo
from ..tools.misc import findnearest, cmdsplit
from math import cos, atan2, radians, degrees


def radarclick(cmdline, lat, lon, traf, navdb):
    """Process lat,lon as clicked in radar window"""
    tostack   = ""
    todisplay = ""

    # Specify which argument can be clicked, and how, in this dictionary
    # and when it's the last, also add ENTER

    clickcmd = {"": "acid,-",
                "POS": "acid",
                "SSD": "acid",
                "CRE":  "-,-,latlon,-,hdg,-,-",
                "HDG": "acid,hdg",
                "SPD": "acid,-",
                "ALT": "acid,-",
                "LISTRTE": "acid,-",
                "ADDWPT": "acid,latlon,-,-,-",
                "ASAS": "acid,-",
                "DEL": "acid,-",
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
                "LINE": "-,latlon,-,latlon",
                "AREA": "latlon,-,latlon",
                "BOX": "-,latlon,-,latlon",
                "POLY": "-,latlon,...",
                "POLYGON": "-,latlon,...",
                "CIRCLE": "-,latlon,-,dist"
                }

    # Split command line into command and arguments, pass traf ids to check for
    # switched acid and command
    cmd, args = cmdsplit(cmdline, traf.id)
    numargs   = len(args)

    # -------- Process click --------
    # Double click on aircraft = POS command
    if numargs == 0 and traf.id.count(cmd) > 0:
        todisplay = "\n"          # Clear the current command
        tostack   = "POS " + cmd  # And send a pos command to the stack

    # Insert: nearest aircraft id
    else:
        # Try to find command in clickcmd dictionary
        try:
            lookup = clickcmd[cmd]

        except KeyError:
            # When command was not found in dictionary:
            # do nothing, return empty strings
            return "", ""

        # For valid value, insert relevant dat on edit line
        if lookup:
            if len(cmdline) > 0 and cmdline[-1] != " ":
                todisplay = " "

            # Determine argument click type
            clickargs = lookup.lower().split(",")
            totargs   = len(clickargs)
            curarg    = numargs
            # Exception case: if the last item of the clickargs list is "..."
            # then the one-but-last can be repeatedly added
            # (e.g. for the definition of a polygon)

            if clickargs[-1] == "...":
                totargs = 999
                curarg  = min(curarg, len(clickargs) - 2)

            if curarg < totargs:
                clicktype = clickargs[curarg]

                if clicktype == "acid":
                    idx = findnearest(lat, lon, traf.lat, traf.lon)
                    if idx >= 0:
                        todisplay += traf.id[idx] + " "

                elif clicktype == "latlon":
                    todisplay += str(round(lat, 6)) + "," + str(round(lon, 6)) + " "

                elif clicktype == "dist":
                    latref, lonref = float(args[1]), float(args[2])
                    todisplay += str(round(geo.kwikdist(latref, lonref, lat, lon), 6))

                elif clicktype == "apt":
                    idx = findnearest(lat, lon, navdb.aplat, navdb.aplon)
                    if idx >= 0:
                        todisplay += navdb.apid[idx] + " "

                elif clicktype == "hdg":
                    # Read start position from command line
                    if cmd == "CRE":
                        try:
                            reflat = float(args[2])
                            reflon = float(args[3])
                            synerr = False
                        except:
                            synerr = True
                    elif cmd == "MOVE":
                        try:
                            reflat = float(args[1])
                            reflon = float(args[2])
                            synerr = False
                        except:
                            synerr = True
                    else:
                        if traf.id.count(args[0]) > 0:
                            idx    = traf.id.index(args[0])
                            reflat = traf.lat[idx]
                            reflon = traf.lon[idx]
                            synerr = False
                        else:
                            synerr = True
                    if not synerr:
                        dy = lat - reflat
                        dx = (lon - reflon) * cos(radians(reflat))
                        hdg = degrees(atan2(dx, dy)) % 360.

                        todisplay += str(int(hdg)) + " "

                # Is it the last argument? (then we will insert ENTER as well)
                if curarg + 1 >= totargs:
                    tostack = cmdline + todisplay
                    todisplay = todisplay + '\n'

    return tostack, todisplay
