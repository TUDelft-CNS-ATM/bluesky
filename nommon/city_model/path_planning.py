#!/usr/bin/python

"""
A module to compute an optimal route from origin to destination
"""
from nommon.city_model.utils import nearestNode3d

from networkx.algorithms.shortest_paths.weighted import single_source_dijkstra

import networkx as nx
import osmnx as ox


__author__ = 'jbueno'
__copyright__ = '(c) Nommon 2021'



def trajectoryCalculation( G, orig, dest ):
    """
    Calculate a optimal trajectory between two points. It computes the nearest nodes to both point
    and the optimal trajectory is computed based on some parameter. The default parameter is
    weight = "travel_time", but we can use weigth = "length".

    Args:
            G (graph): a graph representing the city
            orig (list): with the coordinates of the origin point [longitude, latitude]
            dest (list): with the coordinates of the destination point [longitude, latitude]

    Returns:
            weight (float): float indicating the value of the objetive function
            route (list): list containing the waypoints of the optimal route.
    """
    print( 'Calculating the route...' )
    # Origin
    if len( orig ) == 2:
        orig_node = ox.distance.nearest_nodes( G, X=orig[0], Y=orig[1] )
    elif len( orig ) == 3:
        orig_node = nearestNode3d( G, lon=orig[0], lat=orig[1], altitude=orig[2] )
    elif len( orig ) == 1:
        raise ValueError( 'Origin node needs at least 2 values' )
    else:
        raise ValueError( 'Origin node has too many values' )
    # Destination
    if len( dest ) == 2:
        dest_node = ox.distance.nearest_nodes( G, X=dest[0], Y=dest[1] )
    elif len( dest ) == 3:
        dest_node = nearestNode3d( G, lon=dest[0], lat=dest[1], altitude=dest[2] )
    elif len( dest ) == 1:
        raise ValueError( 'Destination node needs at least 2 values' )
    else:
        raise ValueError( 'Destination node has too many values' )

    # find the shortest path between nodes, minimizing travel time
    weight, route = single_source_dijkstra( G, source=orig_node, target=dest_node,
                                            weight='travel_time' )
    return weight, route


def printRoute( G, route ):
    """
    Print the route

    Args:
            G (graph): a graph representing the city
            route (list): list containing the waypoints of the optimal route.
    """
    print( 'Printing the route...' )
    print( route )
    fig, ax = ox.plot_graph_route( G, route, node_size=0 )
    return fig, ax

# def getRouteTimes( route ):
#     '''
#     Given a route (as a list of nodes), it returns the time for passing at each waypoint
#     The speed considered is the maximum speed of the edges
#     '''
#     index = 0
#     route_times = []
#     for point in route:
#         if index == 0:
#             point_prev = point
#             route_times.append( 0.0 )
#             index += 1
#         else:
#             route_times.append( route_times[-1] + G.edges[ point_prev, point, 0 ]['travel_time'] )
#             # route_times.append( G.edges[ point_prev, point, 0 ]['travel_time'] ) # Sequential
#             point_prev = point
#             index += 1
#
#     return dict( zip( route, route_times ) )
#
# def checkIfConflict( past_routes, past_takeoff_times, new_route, new_takeoff_time ):
#     for old_route in past_routes:
#         pass
#     return None

if __name__ == '__main__':
    pass
