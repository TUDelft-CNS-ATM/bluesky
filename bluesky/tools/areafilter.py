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
    if areaname == 'LIST':
        if not areas:
            return True, 'No shapes are currently defined.'
        else:
            return True, 'Currently defined shapes:\n' + \
                ', '.join(areas)
    if not coordinates:
        if areaname in areas:
            return True, str(areas[areaname])
        else:
            return False, f'Unknown shape: {areaname}'
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
        self.name = name
        self.coordinates = coordinates
        self.top = np.maximum(bottom, top)
        self.bottom = np.minimum(bottom, top)

    def checkInside(self, lat, lon, alt):
        return False

    def _str_vrange(self):
        if self.top < 9e8:
            if self.bottom > -9e8:
                return f' with altitude between {self.bottom} and {self.top}'
            else:
                return f' with altitude below {self.top}'
        if self.bottom > -9e8:
            return f' with altitude above {self.bottom}'
        return ''

    def __str__(self):
        return f'{self.name} is a {self.raw["shape"]} with coordinates ' + \
            ', '.join(str(c) for c in self.coordinates) + self._str_vrange()


class Line(Shape):
    def __init__(self, name, coordinates):
        super().__init__('LINE', name, coordinates)

    def __str__(self):
        return f'{self.name} is a LINE with ' \
            f'start point ({self.coordinates[0]}, {self.coordinates[1]}), ' \
            f'and end point ({self.coordinates[2]}, {self.coordinates[3]}).'


class Box(Shape):
    def __init__(self, name, coordinates, top=1e9, bottom=-1e9):
        super().__init__('BOX', name, coordinates, top, bottom)
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
        super().__init__('CIRCLE', name, coordinates, top, bottom)
        self.clat   = coordinates[0]
        self.clon   = coordinates[1]
        self.r      = coordinates[2]

    def checkInside(self, lat, lon, alt):
        distance = kwikdist(self.clat, self.clon, lat, lon)  # [NM]
        inside   = (distance <= self.r) & (self.bottom <= alt) & (alt <= self.top)
        return inside

    def __str__(self):
        return f'{self.name} is a CIRCLE with ' \
            f'center ({self.clat}, {self.clon}) ' \
            f'and radius {self.r}.' + self._str_vrange()


class Poly(Shape):
    def __init__(self, name, coordinates, top=1e9, bottom=-1e9):
        super().__init__('POLY', name, coordinates, top, bottom)
        self.border = Path(np.reshape(coordinates, (len(coordinates) // 2, 2)))

    def checkInside(self, lat, lon, alt):
        points = np.vstack((lat,lon)).T
        inside = np.all((self.border.contains_points(points), self.bottom <= alt, alt <= self.top), axis=0)
        return inside
