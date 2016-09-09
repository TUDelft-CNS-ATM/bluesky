"""Area filter module"""
from geo import kwikdist
from matplotlib.path import Path
import numpy as np


areas = dict()


def listACInside(self, areaname, traf, acids=None):
    if areaname not in areas:
        return set()
    if acids is None:
        acidx = range(traf.ntraf)
    else:
        acidx = [traf.id2idx(i) for i in acids]

    area = areas[areaname]
    return area.inside(traf, acidx)


def listACOutside(areaname, traf, acids=None):
    acinside = listACInside(areaname, traf, acids)
    return set(range(traf.ntraf)) - acinside


def defineArea(scr, areaname, areatype, coordinates):
    # When top is skipped in stack, None is entered instead. Replace with 1e9
    if coordinates[-2] is None:
        coordinates[-2] = 1e9

    if areatype == 'BOX':
        areas[areaname] = Box(coordinates[:4], *coordinates[4:])
    elif areatype == 'CIRCLE':
        areas[areaname] = Circle(coordinates[:2], *coordinates[2:])
    elif areatype == 'POLY':
        areas[areaname] = Poly(coordinates)
    elif areatype == 'POLYALT':
        areas[areaname] = Poly(coordinates[2:], *coordinates[:2])

    # Pass the shape on to the screen object
    scr.objappend(areatype, areaname, coordinates)


def deleteArea(scr, areaname):
    if areaname in areas:
        areas.pop(areaname)
        scr.objappend('', areaname, None)
    pass


def reset():
    areas.clear()


class Box:
    def __init__(self, coordinates, top=1e9, bottom=-1e9):
        self.lat0, self.lon0, self.lat1, self.lon1 = coordinates
        self.top    = top
        self.bottom = bottom

    def inside(self, traf, acidx):
        ret = []
        for i in acidx:
            if self.lat0 <= traf.lat[i] <= self.lat1 and \
               self.lon0 <= traf.lon[i] <= self.lon1 and \
               self.bottom <= traf.alt[i] <= self.top:
                ret.append(i)

        return ret
        # What to do with swtaxi?
        # (traf.alt[i] >= 0.5*ft or traf.swtaxi)


class Circle:
    def __init__(self, center, radius, top=1e9, bottom=-1e9):
        self.clat   = center[0]
        self.clon   = center[1]
        self.r      = radius
        self.top    = top
        self.bottom = bottom

    def inside(self, traf, acidx):
        ret = []
        # delete aircraft if it is too far from the center of the circular area, or if has decended below the minimum altitude
        for i in acidx:
            distance = kwikdist(self.clat0, self.clon0, traf.lat[i], traf.lon[i])  # [NM]
            if distance <= self.r and self.bottom <= traf.alt[i] <= self.top:
                ret.append(i)
        return ret


class Poly:
    def __init__(self, coordinates, top=1e9, bottom=-1e9):
        self.border = Path(np.reshape(coordinates, (len(coordinates) / 2, 2)), closed=True)
        self.top    = top
        self.bottom = bottom

    def inside(self, traf, acidx):
        ret = []
        for i in acidx:
            if self.border.contains_point([traf.lat[i], traf.lon[i]]) and \
               self.bottom <= traf.alt[i] <= self.top:
                ret.append(i)
        return ret
