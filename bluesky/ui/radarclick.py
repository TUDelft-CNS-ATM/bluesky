from math import cos, atan2, radians, degrees
from numpy import array
import bluesky as bs

from bluesky.tools import geo
from bluesky.tools.misc import findnearest, cmdsplit


def radarclick(cmdline, lat, lon, acdata=None, route=None):
    """Process lat,lon as clicked in radar window"""

    # Specify which argument can be clicked, and how, in this dictionary
    # and when it's the last, also add ENTER

    clickcmd = {"": "acid,-",
                "ADDWPT": "acid,latlon,-,-,wpinroute,-",
                "AFTER": "acid,wpinroute,-",
                "AT": "acid,wpinroute,-",
                "ALT": "acid,-",
                "AREA": "latlon,-,latlon",
                "ASAS": "acid,-",
                "BOX": "-,latlon,-,latlon",
                "CIRCLE": "-,latlon,-,dist",
                "CRE":  "-,-,latlon,-,hdg,-,-",
                "DEFWPT": "-,latlon,-",
                "DEL": "acid,...",
                "DELWPT": "acid,wpinroute,-",
                "DELRTE": "acid,-",
                "DEST": "acid,apt",
                "DIRECT": "acid,wpinroute",
                "DIST": "latlon,-,latlon",
                "DUMPRTE": "acid",
                "ENG": "acid,-",
                "GETWIND":"latlon,-",
                "GROUP":"-,acid,...",
                "HDG": "acid,hdg",
                "LINE": "-,latlon,-,latlon",
                "LISTRTE": "acid,-",
                "LNAV": "acid,-",
                "MOVE": "acid,latlon,-,-,hdg",
                "NAVDISP": "acid",
                "NOM": "acid",
                "ND": "acid",
                "ORIG": "acid,apt",
                "PAN": "latlon",
                "POLY": "-,latlon,...",
                "POLYALT": "-,-,-,latlon,...",
                "POLYGON": "-,latlon,...",
                "POLYLINE": "-,latlon,...",
                "POS": "acid",
                "SSD": "acid,...",
                "SPD": "acid,-",
                "TRAIL":"acid,-",
                "VNAV": "acid,-",
                "VS": "acid,-",
                "WIND":"latlon,-",
                "WINDGFS":"latlon,-,latlon,-"
                }

    # Default values, when nothing is found to be added based on click
    todisplay = ""  # Result of click is added here
    tostack   = ""  # If it is the last argument we will pass whole line to the stack

    # The pygame version has access to the complete traffic object. This gets
    # passed to radarclick in the QtGL version.
    if acdata is None:
        acdata = bs.traf

    # Split command line into command and arguments, pass traf ids to check for
    # switched acid and command
    cmd, args = cmdsplit(cmdline, acdata.id)
    cmd = cmd.upper()
    numargs   = len(args)

    # -------- Process click --------
    # Double click on aircraft = POS command
    if numargs == 0 and acdata.id.count(cmd.upper()) > 0:
        todisplay = "\n"          # Clear the current command
        tostack   = "POS " + cmd  # And send a pos command to the stack

    # Insert: nearest aircraft id
    else:

        # TODO: Check for synonyms (dictionary is imported from stack)
        # if cmd in cmdsynon:
        #    cmd = cmdsynon[cmd.upper()]

        # Try to find command in clickcmd dictionary
        try:
            lookup = clickcmd[cmd.upper()]

        except KeyError:
            # When command was not found in dictionary:
            # do nothing, return empty strings
            return "", ""

        # For valid value, insert relevant dat on edit line
        if lookup:
            if len(cmdline) > 0 and (cmdline[-1] != " " and cmdline[-1]!=","):
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
                    idx = findnearest(lat, lon, acdata.lat, acdata.lon)
                    if idx >= 0:
                        todisplay += acdata.id[idx] + " "

                elif clicktype == "latlon":
                    todisplay += str(round(lat, 6)) + "," + str(round(lon, 6)) + " "

                elif clicktype == "dist":
                    latref, lonref = float(args[1]), float(args[2])
                    todisplay += str(round(geo.kwikdist(latref, lonref, lat, lon), 6))

                elif clicktype == "apt":
                    idx = findnearest(lat, lon, bs.navdb.aptlat, bs.navdb.aptlon)
                    if idx >= 0:
                        todisplay += bs.navdb.aptid[idx] + " "

                elif clicktype == "wpinroute":  # Find nearest waypoint in route
                    if acdata.id.count(args[0]) > 0:
                        itraf      = acdata.id.index(args[0])
                        synerr = False
                        reflat = acdata.lat[itraf]
                        reflon = acdata.lon[itraf]
                        # The pygame version can get the route directly from traf
                        # otherwise the route is passed to this function
                        if route is None:
                            route = acdata.ap.route[itraf]

                        if len(route.wplat) > 0:
                            iwp = findnearest(lat, lon,
                                        array(route.wplat),
                                        array(route.wplon))
                            if iwp >= 0:
                                todisplay += route.wpname[iwp]+" "

                    else:
                        synerr = True

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
                        if acdata.id.count(args[0]) > 0:
                            idx    = acdata.id.index(args[0])
                            reflat = acdata.lat[idx]
                            reflon = acdata.lon[idx]
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
                    # todisplay = todisplay + '\n'
                    todisplay = ''

    return tostack, todisplay
