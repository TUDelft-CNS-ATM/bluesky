#!/usr/bin/python

"""

"""
from networkx.algorithms.shortest_paths.weighted import single_source_dijkstra

__author__ = 'jbueno'
__copyright__ = '(c) Nommon 2021'

import networkx as nx
import osmnx as ox


def trajectoryCalculation( G, orig, dest ):
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


if __name__ == '__main__':
    pass
