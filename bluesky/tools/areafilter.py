"""Area filter module"""
from matplotlib.path import Path
import numpy as np
import bluesky as bs
from bluesky.tools.geo import kwikdist

areas = dict()


def hasArea(areaname):
    """Check if area with name 'areaname' exists."""
    return areaname in areas


def defineArea(areaname, areatype, coordinates, top=1e9, bottom=-1e9):
    """Define a new area"""
    if areatype == 'BOX':
        areas[areaname] = Box(areaname, coordinates, top, bottom)
    elif areatype == 'CIRCLE':
        areas[areaname] = Circle(areaname, coordinates, top, bottom)
    elif areatype[:4] == 'POLY':
        areas[areaname] = Poly(areaname, coordinates, top, bottom)
    elif areatype == 'LINE':
        areas[areaname] = Line(areaname, coordinates)

    # Pass the shape on to the screen object
    bs.scr.objappend(areatype, areaname, coordinates)

def checkInside(areaname, lat, lon, alt):
    """ Check if points with coordinates lat, lon, alt are inside area with name 'areaname'.
        Returns an array of booleans. True ==  Inside"""
    if areaname not in areas:
        return np.zeros(len(lat), dtype=np.bool)
    area = areas[areaname]
    return area.checkInside(lat, lon, alt)

def deleteArea(areaname):
    """ Delete area with name 'areaname'. """
    if areaname in areas:
        areas.pop(areaname)
        bs.scr.objappend('', areaname, None)

def reset():
    """ Clear all data. """
    areas.clear()

class Shape:
    def __init__(self, shape, name, coordinates, top=1e9, bottom=-1e9):
        self.raw = dict(name=name, shape=shape, coordinates=coordinates)
        self.coordinates = coordinates
        self.top = np.maximum(bottom, top)
        self.bottom = np.minimum(bottom, top)

    def checkInside(self, lat, lon, alt):
        return False


class Line(Shape):
    def __init__(self, name, coordinates):
        super(Line, self).__init__('LINE', name, coordinates)


class Box(Shape):
    def __init__(self, name, coordinates, top=1e9, bottom=-1e9):
        super(Box, self).__init__('BOX', name, coordinates, top, bottom)
        # Sort the order of the corner points
        self.lat0 = min(coordinates[0], coordinates[2])
        self.lon0 = min(coordinates[1], coordinates[3])
        self.lat1 = max(coordinates[0], coordinates[2])
        self.lon1 = max(coordinates[1], coordinates[3])

    def checkInside(self, lat, lon, alt):
        return ((self.lat0 <=  lat) & ( lat <= self.lat1)) & \
               ((self.lon0 <= lon) & (lon <= self.lon1)) & \
               ((self.bottom <= alt) & (alt <= self.top))



class Circle(Shape):
    def __init__(self, name, coordinates, top=1e9, bottom=-1e9):
        super(Circle, self).__init__('CIRCLE', name, coordinates, top, bottom)
        self.clat   = coordinates[0]
        self.clon   = coordinates[1]
        self.r      = coordinates[2]

    def checkInside(self, lat, lon, alt):
        distance = kwikdist(self.clat, self.clon, lat, lon)  # [NM]
        inside   = (distance <= self.r) & (self.bottom <= alt) & (alt <= self.top)
        return inside


class Poly(Shape):
    def __init__(self, name, coordinates, top=1e9, bottom=-1e9):
        super(Poly, self).__init__('POLY', name, coordinates, top, bottom)
        self.border = Path(np.reshape(coordinates, (len(coordinates) // 2, 2)))

    def checkInside(self, lat, lon, alt):
        points = np.vstack((lat,lon)).T
        inside = np.all((self.border.contains_points(points), self.bottom <= alt, alt <= self.top), axis=0)
        return inside
