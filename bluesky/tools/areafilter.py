"""Area filter module"""

import numpy as np

from bluesky.tools import shapes
from bluesky.stack import commandgroup
from bluesky.network.publisher import StatePublisher


# Dictionary of all basic shapes (The shape classes defined in this file) by name
basic_shapes = dict()
# Publisher object to manage publishing of states to clients
polypub = StatePublisher('POLY', collect=True)


@polypub.payload
def puball():
    return dict(polys={name: poly.raw for name, poly in basic_shapes.items()})


@commandgroup(annotations='txt,color', aliases=('COLOR', 'COL'))
def colour(name, r, g, b):
    ''' Set custom color for visual objects. '''
    poly = basic_shapes.get(name)
    if poly:
        poly.color = (r, g, b)
        polypub.send_update(polys={name:dict(color=poly.color)})
        return True
    return False, 'No shape found with name ' + name


def hasArea(areaname):
    """Check if area with name 'areaname' exists."""
    return areaname in basic_shapes


def getArea(areaname):
    ''' Return the area object corresponding to name '''
    return basic_shapes.get(areaname, None)


def defineArea(name, shape, coordinates, top=1e9, bottom=-1e9):
    """Define a new area"""
    if name == 'LIST':
        if not basic_shapes:
            return True, 'No shapes are currently defined.'
        else:
            return True, 'Currently defined shapes:\n' + \
                ', '.join(basic_shapes)
    if coordinates is None:
        if name in basic_shapes:
            return True, str(basic_shapes[name])
        else:
            return False, f'Unknown shape: {name}'
    if shape == 'BOX':
        basic_shapes[name] = shapes.Box(name, coordinates, top, bottom)
    elif shape == 'CIRCLE':
        basic_shapes[name] = shapes.Circle(name, coordinates, top, bottom)
    elif shape[:4] == 'POLY':
        basic_shapes[name] = shapes.Poly(name, coordinates, top, bottom)
    elif shape == 'LINE':
        basic_shapes[name] = shapes.Line(name, coordinates)

    # Pass the shape on to the connected clients
    polypub.send_update(polys={name:dict(shape=shape, coordinates=coordinates)})

    return True  #, f'Created {shape} {name}'


def checkInside(areaname, lat, lon, alt):
    """ Check if points with coordinates lat, lon, alt are inside area with name 'areaname'.
        Returns an array of booleans. True ==  Inside"""
    if areaname not in basic_shapes:
        return np.zeros(len(lat), dtype=bool)
    area = basic_shapes[areaname]
    return area.checkInside(lat, lon, alt)

def deleteArea(areaname):
    """ Delete area with name 'areaname'. """
    if areaname in basic_shapes:
        basic_shapes.pop(areaname)
        # bs.scr.objappend('', areaname, None)
        polypub.send_delete(polys=[areaname])

def reset():
    """ Clear all data. """
    basic_shapes.clear()
    shapes.Shape.reset()


def get_intersecting(lat0, lon0, lat1, lon1):
    ''' Return all shapes that intersect with a specified rectangular area.

        Arguments:
        - lat0/1, lon0/1: Coordinates of the top-left and bottom-right corner
          of the intersection area.
    '''
    items = shapes.Shape.areatree.intersection((lat0, lon0, lat1, lon1))
    return [shapes.Shape.areas_by_id[i.id] for i in items]


def get_knearest(lat0, lon0, lat1, lon1, k=1):
    ''' Return the k nearest shapes to a specified rectangular area.

        Arguments:
        - lat0/1, lon0/1: Coordinates of the top-left and bottom-right corner
          of the relevant area.
        - k: The (maximum) number of results to return.
    '''
    items = shapes.Shape.areatree.nearest((lat0, lon0, lat1, lon1), k)
    return [shapes.Shape.areas_by_id[i.id] for i in items]



