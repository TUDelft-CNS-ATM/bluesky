''' BlueSky Shapes '''
import numpy as np
from matplotlib.path import Path
from bluesky.tools.geo import kwikdist

from weakref import WeakValueDictionary
try:
    from rtree.index import Index
except (ImportError, OSError):
    print('Warning: RTree could not be loaded. areafilter get_intersecting and get_knearest won\'t work')
    class Index:
        ''' Dummy index class for installations where rtree is missing
            or doesn't work.
        '''
        @staticmethod
        def intersection(*args, **kwargs):
            return []

        @staticmethod
        def nearest(*args, **kwargs):
            return []

        @staticmethod
        def insert(*args, **kwargs):
            return

        @staticmethod
        def delete(*args, **kwargs):
            return


class Shape:
    '''
        Base class of BlueSky shapes
    '''
    # Global counter to keep track of used shape ids
    max_area_id = 0

    # Weak-value dictionary of all Shape-derived objects by name, and id
    areas_by_id = WeakValueDictionary()
    areas_by_name = WeakValueDictionary()

    # RTree of all areas for efficient geospatial searching
    areatree = Index()

    @classmethod
    def reset(cls):
        ''' Reset shape data when simulation is reset. '''
        # Weak dicts and areatree should be cleared automatically
        # Reset max area id
        cls.max_area_id = 0

    def __init__(self, name, coordinates, top=1e9, bottom=-1e9):
        self.name = name
        self.coordinates = coordinates
        self.top = np.maximum(bottom, top)
        self.bottom = np.minimum(bottom, top)
        lat = coordinates[::2]
        lon = coordinates[1::2]
        self.bbox = [min(lat), min(lon), max(lat), max(lon)]

        # Global weak reference and tree storage
        self.area_id = Shape.max_area_id
        Shape.max_area_id += 1
        Shape.areas_by_id[self.area_id] = self
        Shape.areas_by_name[self.name] = self
        Shape.areatree.insert(self.area_id, self.bbox)

    def __del__(self):
        # Objects are removed automatically from the weak-value dicts,
        # but need to be manually removed from the rtree
        Shape.areatree.delete(self.area_id, self.bbox)

    @property
    def raw(self):
        ret = dict(name=self.name, shape=self.kind(), coordinates=self.coordinates)
        if hasattr(self, 'color'):
            ret['color'] = self.color
        return ret

    def checkInside(self, lat, lon, alt):
        ''' Returns True (or boolean array) if coordinate lat, lon, alt lies
            within this shape.

            Reimplement this function in the derived shape classes for this to
            work.
        '''
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

    @classmethod
    def kind(cls):
        ''' Return a string describing what kind of shape this is. '''
        return cls.__name__.upper()


class Line(Shape):
    ''' A line shape '''
    def __init__(self, name, coordinates):
        super().__init__(name, coordinates)

    def __str__(self):
        return f'{self.name} is a LINE with ' \
            f'start point ({self.coordinates[0]}, {self.coordinates[1]}), ' \
            f'and end point ({self.coordinates[2]}, {self.coordinates[3]}).'


class Box(Shape):
    ''' A box shape '''
    def __init__(self, name, coordinates, top=1e9, bottom=-1e9):
        super().__init__(name, coordinates, top, bottom)
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
    ''' A circle shape '''
    def __init__(self, name, coordinates, top=1e9, bottom=-1e9):
        super().__init__(name, coordinates, top, bottom)
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
    ''' A polygon shape '''
    def __init__(self, name, coordinates, top=1e9, bottom=-1e9):
        super().__init__(name, coordinates, top, bottom)
        self.border = Path(np.reshape(coordinates, (len(coordinates) // 2, 2)))

    def checkInside(self, lat, lon, alt):
        points = np.vstack((lat,lon)).T
        inside = np.all((self.border.contains_points(points), self.bottom <= alt, alt <= self.top), axis=0)
        return inside


def get_by_name(name: str) -> Shape|None:
    ''' Get an existing shape by name. Returns None if shape doesn't exist. '''
    return Shape.areas_by_name.get(name)
