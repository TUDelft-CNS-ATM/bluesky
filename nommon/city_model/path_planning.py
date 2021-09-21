#!/usr/bin/python

"""

"""
from networkx.algorithms.shortest_paths.weighted import single_source_dijkstra

__author__ = 'jbueno'
__copyright__ = '(c) Nommon 2021'

import networkx as nx
import osmnx as ox


def trajectoryCalculation( G, orig, dest ):
    print( 'Calculating the route...' )
    orig_node = ox.distance.nearest_nodes( G, X=orig[0], Y=orig[1] )
    dest_node = ox.distance.nearest_nodes( G, X=dest[0], Y=dest[1] )

    # find the shortest path between nodes, minimizing travel time, then plot it
#     route = nx.shortest_path( G, source=orig_node, target=dest_node, weight='travel_time' )
    length, route = single_source_dijkstra( G, source=orig_node, target=dest_node,
                                            weight='travel_time' )
    return length, route


def printRoute( G, route ):
    print( 'Printing the route...' )
    print( route )
    fig, ax = ox.plot_graph_route( G, route, node_size=0 )
    return fig, ax

def getRouteTimes( route ):
    '''
    Given a route (as a list of nodes), it returns the time for passing at each waypoint
    The speed considered is the maximum speed of the edges
    '''
    index = 0
    route_times = []
    for point in route:
        if index == 0:
            point_prev = point
            route_times.append( 0.0 )
            index += 1
        else:
            route_times.append( route_times[-1] + G.edges[ point_prev, point, 0 ]['travel_time'] )
            # route_times.append( G.edges[ point_prev, point, 0 ]['travel_time'] ) # Sequential
            point_prev = point
            index += 1

    return dict( zip( route, route_times ) )

def checkIfConflict( past_routes, past_takeoff_times, new_route, new_takeoff_time ):
    for old_route in past_routes:
        pass
    return None

if __name__ == '__main__':

    import os
    import configparser

    from auxiliar import read_my_graphml
    from city_graph import cityGraph
    from multi_di_graph_3D import MultiDiGrpah3D

    import numpy as np

    # CONFIG
    config = configparser.ConfigParser()
    config_path = "C:/workspace3/bluesky/nommon/city_model/settings.cfg"
    config.read( config_path )

    # City
    if config['City'].getboolean( 'import' ):
        filepath = r"C:\workspace3\bluesky\nommon\city_model\data\hannover_segments.graphml"
        G = read_my_graphml( filepath )
        G = MultiDiGrpah3D( G )
#         fig, ax = ox.plot_graph( G )
        segments = np.load( 'my_segments.npy', allow_pickle='TRUE' ).item()
    else:
        G = cityGraph( config )
    # Route
    orig = [9.72321, 52.3761]
    dest = [9.766, 52.39 ]
    length, route = trajectoryCalculation( G, orig, dest )

    times = getRouteTimes( route )

    print( route )
    print( times )

