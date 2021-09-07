#!/usr/bin/python

"""

"""


__author__ = 'mbaena'
__copyright__ = '(c) Nommon 2021'


from dynamic_segments import dynamicSegments


def isContained( point, box ):
    """
    Returns True if the point is contained in the box

    Input
        point - list with 2 variables [lon, lat]
        box - list with 4 variables [lon_min, lon_max, lat_min, lat_max]
    """
    if ( point[0] > box[0] ) and ( point[0] < box[1] ):
        if ( point[1] > box[2] ) and ( point[1] < box[3] ):
            return True
        else:
            return False
    else:
        return False

def restrictedSegments( G, segments, coordinates, speed, capacity, config ):
    """
    This function imposes speed and capacity limitations to the desired segments

    # Segments attributes
    # {'lon_min': 9.75, 'lon_max': 9.765,
        'lat_min': 52.375, 'lat_max': 52.3875,
        'z_min': 0.0, 'z_max': 125.0,
        'speed': 39.0, 'capacity': 16,
        'new': False, 'updated': False}

    Input
        G
        segments
        coordinates - point, polygon defining the restriction zone ([lon, lat])
        speed - speed limitation
        capacity - capacity limitation

    Output
        G
        segments

    NOTE: THIS WOULD NOT WORK PROPOERLY IF THE RESTRICTION ZONE IS TOO LARGE. AN INTERIOR SEGMENT
        MAY NOT BE TAKEN INTO ACCOUNT
    """
    for segment in segments:
        box = [ segments[segment]['lon_min'] , segments[segment]['lon_max'],
               segments[segment]['lat_min'], segments[segment]['lat_max'] ]
        for coordinate in coordinates:
            if isContained( coordinate, box ):
                segments[segment]['speed'] = 0
                segments[segment]['updated'] = True

    G, segments = dynamicSegments( G, config, segments )
    return G, segments

if __name__ == '__main__':
    point_contained = [1, 1]
    point_not_contained = [3, 4]
    box = [0, 2, 0, 2]

    print( 'Contained:', isContained( point_contained, box ) )
    print( 'Not Contained:', isContained( point_not_contained, box ) )
