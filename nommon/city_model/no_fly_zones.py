#!/usr/bin/python

"""

"""


__author__ = 'mbaena'
__copyright__ = '(c) Nommon 2021'


from nommon.city_model.dynamic_segments import dynamicSegments

from shapely.geometry import Polygon


def intersection( restricted_area, segment ):
    """
    Returns True if the point is contained in the box

    Input:
        restricted_area - shape of the restricted area.
                            Polygon defined by its edges  [(lon1, lat1), (lon2, lat2)...]
        segment - segment. Only segment coordinates are used [lon_min, lon_max, lat_min, lat_max]

    Output:
        boolean - true if restricted area intersects the segment area
    """

    restricted_area_shape = Polygon( restricted_area )
    segment_box = [( segment[segment]['lon_min'], segment[segment]['lat_min'] ),
                   ( segment[segment]['lon_max'], segment[segment]['lat_min'] ),
                   ( segment[segment]['lon_max'], segment[segment]['lat_max'] ),
                   ( segment[segment]['lon_min'], segment[segment]['lat_max'] ),
                   ( segment[segment]['lon_min'], segment[segment]['lat_min'] )]

    segmet_shape = Polygon( segment_box )
    return restricted_area_shape.intersects( segmet_shape )


def restrictedSegments( G, segments, restricted_area, speed, capacity, config ):
    """
    This function imposes speed and capacity limitations to the desired segments

    # Segments attributes example:
    # {'lon_min': 9.75, 'lon_max': 9.765,
        'lat_min': 52.375, 'lat_max': 52.3875,
        'z_min': 0.0, 'z_max': 125.0,
        'speed': 39.0, 'capacity': 16,
        'new': False, 'updated': False}

    Input
        G - graph
        segments - segments
        restricted_area - point, polygon defining the restriction zone [(lon1, lat1), (lon2, lat2)...]
        speed - speed limitation
        capacity - capacity limitation

    Output
        G - updated graph with restricted zones
        segments - updated segments with restricted zones
    """
    for segment in segments:
        if intersection( restricted_area, segment ):
            segments[segment]['speed'] = 0
            segments[segment]['updated'] = True

    G, segments = dynamicSegments( G, config, segments )
    return G, segments


if __name__ == '__main__':
    pass
