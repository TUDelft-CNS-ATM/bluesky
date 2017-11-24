# -*- coding: utf-8 -*-

import bluesky as bs
from .misc import txt2lat, txt2lon


def txt2pos(name, reflat, reflon):
    """
    Converts a 'name' (lat/lon, wpt, apt, etc) into a position.
    Returns the tuple:
    - True, Position object if successful
    - False, Reason if unsuccessful
    """

    pos = Position(name.upper().strip(), reflat, reflon)
    if not pos.error:
        return True, pos
    else:
        return False, name + " not found in database"


def islat(txt):
    """
    Determines whether 'txt' looks like a latitude value.
    """
    return islatlon(txt, ["N", "S"])


def islon(txt):
    """
    Determines whether 'txt' looks like a longitude value.
    """
    return islatlon(txt, ["E", "W"])


def islatlon(txt, dirs):
    """
    Determines whether the passed value looks like a lat or lon value,
    or a compound lat,lon value.
    """

    # Take out non-digit chars which are allowed,  We split in case this is
    # a compound lat/lon value.
    #
    testtxt = txt.upper().\
        strip(" -+\n").\
        split(",")[0].replace('"', "").replace("'", "").replace(".", "")

    # Take away one leading N, S / E, W if present before other chars
    if testtxt[0] in dirs and len(testtxt) > 1:
        testtxt = testtxt[1:]

    try:
        float(testtxt)
    except ValueError:
        return False
    return True


class Position:

    """
    Position class: container for position data
    """

    # position types
    latlon = 0  # lat/lon waypoint
    nav = 1  # VOR/nav database waypoint
    apt = 2  # airport
    runway = 3  # runway
    dir = 4

    # Initialize using text
    def __init__(self, name, reflat, reflon):

        self.name = name  # default: copy source name
        self.error = False  # we're optmistic about our succes
        self.lat = self.lon = self.type = None

        # lat,lon type ?
        if name.count(",") > 0:  # lat,lon or apt,rwy type
            txtlat, txtlon = name.split(",")
            if islat(txtlat) and islon(txtlon):
                self.lat = txt2lat(txtlat)
                self.lon = txt2lon(txtlon)
                self.name = ""
                self.type = Position.latlon
            else:
                self.error = True

        # runway type ? "EHAM/RW06","EHGG/RWY27"
        elif name.count("/RW") > 0:
            try:
                aptname, rwytxt = name.split("/RW")
                rwyname = rwytxt.lstrip("Y").upper()  # remove Y and spaces
                self.lat, self.lon = bs.navdb.rwythresholds[
                    aptname][rwyname][:2]  # raises error if not found
            except:
                self.error = True
            self.type = Position.runway

        # airport?
        elif bs.navdb.aptid.count(name) > 0:
            idx = bs.navdb.aptid.index(name.upper())

            self.lat = bs.navdb.aptlat[idx]
            self.lon = bs.navdb.aptlon[idx]
            self.type = Position.apt

        # fix or navaid?
        elif bs.navdb.wpid.count(name) > 0:
            idx = bs.navdb.getwpidx(name, reflat, reflon)
            self.lat = bs.navdb.wplat[idx]
            self.lon = bs.navdb.wplon[idx]
            self.type = Position.nav

        # aircraft id?
        elif bs.traf.id2idx(name) >= 0:
            idx = bs.traf.id2idx(name)
            self.name = ""
            self.type = Position.latlon
            self.lat = bs.traf.lat[idx]
            self.lon = bs.traf.lon[idx]

            # exception for pan, check for LEFT, RIGHT, ABOVE or DOWN
        elif name.upper() in ["LEFT", "RIGHT", "ABOVE", "DOWN"]:
            self.lat = reflat
            self.lon = reflon
            self.type = Position.dir

# Not used now, but save this code for future use
# Make a N52E004 type waypoint name
#            clat = "SN"[lat>0]
#            clon = "WE"[lon>0]
#            name = clat + "%02d"%int(abs(round(lat))) + \
#                   clon + "%03d"%int(abs(round(lon)))
        else:
            self.error = True
            # raise error with missing data... (empty position object)

        return
