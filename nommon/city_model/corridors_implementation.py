#!/usr/bin/python

"""

"""

from dynamic_segments import defineSegment
import osmnx as ox


__author__ = 'jbueno'
__copyright__ = '(c) Nommon 2021'


def insertionNode( G, lon, lat, altitude ):
    nearest_node = ox.distance.nearest_nodes( G, X=lon, Y=lat )

    nodes = list( G.nodes )

    nearest_latlon = list( filter( lambda node: str( node )[1:] == nearest_node[1:], nodes ) )

    nearest_node = nearest_latlon[0]
    delta_z = abs( G.nodes[nearest_node]['z'] - altitude )

    for node in nearest_latlon[1:]:
        delta_z_aux = abs( G.nodes[node]['z'] - altitude )
        if delta_z_aux < delta_z:
            delta_z = delta_z_aux
            nearest_node = node

    return nearest_node


def corridorCreation( G, segments, corridor_coordinates, altitude, speed, capacity, name ):
    segments = defineSegment( segments, 0, 0, 0, 0, 0, 0, speed, capacity, name )

    index = 0
    nodes_G = []
    nodes_corridor = []
    for point in corridor_coordinates:
        index += 1
        nodes_corridor += [name + '_' + str( index )]
        point_lon = point[0]
        point_lat = point[1]

        nodes_G += [insertionNode( G, point_lon, point_lat, altitude )]

        G.addNodeAltitude( nodes_corridor[-1], point_lat, point_lon, altitude )

        if len( nodes_corridor ) == 1:
            continue

        G.add_edge( nodes_corridor[-2], nodes_corridor[-1], 0, oneway=False, segment=name,
                    speed=speed,
                    length=ox.distance.great_circle_vec( point_lat, point_lon,
                                                         G.nodes[nodes_corridor[-2]]['y'],
                                                         G.nodes[nodes_corridor[-2]]['x'] ) )

    for entry_node, corridor_point in zip( nodes_G, nodes_corridor ):
        delta_z = G.nodes[entry_node]['z'] - G.nodes[corridor_point]['z']
        delta_xy = ox.distance.great_circle_vec( G.nodes[entry_node]['y'], G.nodes[entry_node]['x'],
                                                 G.nodes[corridor_point]['y'],
                                                 G.nodes[corridor_point]['x'] )

        length = ( delta_z ** 2 + delta_xy ** 2 ) ** ( 1 / 2 )

        G.add_edge( entry_node, corridor_point, 0, oneway=False, segment='new', speed=50.0,
                    length=length )
        G.add_edge( corridor_point, entry_node, 0, oneway=False, segment=name, speed=speed,
                    length=length )

    return G, segments


if __name__ == '__main__':
    pass
