#!/usr/bin/python

"""

"""
import time

import configparser

from auxiliar import read_my_graphml
from city_graph import cityGraph
from corridors_implementation import corridorCreation, corridorLoad
from dynamic_segments import dynamicSegments
from multi_di_graph_3D import MultiDiGrpah3D
from path_planning import trajectoryCalculation, printRoute
import numpy as np
import osmnx as ox


__author__ = 'jbueno'
__copyright__ = '(c) Nommon 2021'


if __name__ == '__main__':

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

#    fig, ax = ox.plot_graph( G )
    # Segments
#    G, segments = dynamicSegments( G, config, segments=None )
#
#    segments[list( segments.keys() )[0]]['new'] = True
#    segments[list( segments.keys() )[3]]['updated'] = True
#    segments[list( segments.keys() )[3]]['speed'] = 0
#    G, segments = dynamicSegments( G, config, segments )

#    print( 'Saving the graph...' )
#    filepath = "./data/hannover_segments_2.graphml"
#    ox.save_graphml( G, filepath )
#    np.save( 'my_segments_2.npy', segments )

    # Corridors
    G, segments = corridorLoad( G, segments, config )

    # G, segments = corridorCreation( G, segments, ( [9.63055, 52.42053], [9.6478, 52.35128 ], [9.86469, 52.35893], [9.89988, 52.39687], [9.74728, 52.4264] ),
#                                     200, 100, 50, 'COR_1' )
#     G, segments = dynamicSegments( G, config, segments=None )
    # No-fly zones
#     segments['segment_3_1_0']['speed'] = 0
#     segments['segment_3_1_0']['updated'] = True
#     segments['segment_3_1_1']['speed'] = 0
#     segments['segment_3_1_1']['updated'] = True
    segments['segment_2_2_0']['speed'] = 0
    segments['segment_2_2_0']['updated'] = True
    segments['segment_2_2_1']['speed'] = 0
    segments['segment_2_2_1']['updated'] = True
#    segments['segment_1_2_0']['speed'] = 0
#    segments['segment_1_2_0']['updated'] = True
#    segments['segment_1_2_1']['speed'] = 0
#    segments['segment_1_2_1']['updated'] = True
#    segments['segment_1_3_0']['speed'] = 0
#    segments['segment_1_3_0']['updated'] = True
#    segments['segment_1_3_1']['speed'] = 0
#    segments['segment_1_3_1']['updated'] = True
#    segments['segment_2_3_0']['speed'] = 0
#    segments['segment_2_3_0']['updated'] = True
#    segments['segment_2_3_1']['speed'] = 0
#    segments['segment_2_3_1']['updated'] = True
#    segments['segment_3_2_0']['speed'] = 0
#    segments['segment_3_2_0']['updated'] = True
#     segments['segment_3_2_1']['speed'] = 0
#     segments['segment_3_2_1']['updated'] = True
#     segments['segment_3_3_0']['speed'] = 0
#     segments['segment_3_3_0']['updated'] = True
#     segments['segment_3_3_1']['speed'] = 0
#     segments['segment_3_3_1']['updated'] = True
#
    G, segments = dynamicSegments( G, config, segments )
#     edges = ox.utils_graph.graph_to_gdfs( G, nodes=False, fill_edge_geometry=False )
    # Plot graph
    # ec = ["r" if G.edges[( u, v, k )]['speed'] == 0
    #      "b" if G.edges[( u, v, k )]['speed'] == 100
    #      else "gray"
    #      for u, v, k in G.edges( keys=True )]

    ec = []
    for u, v, k in G.edges( keys=True ):
        if G.edges[( u, v, k )]['speed'] == 0:
            ec.append( "r" )
        elif G.edges[( u, v, k )]['speed'] == 100:
            ec.append( "g" )
        else:
            ec.append( "gray" )
    fig, ax = ox.plot_graph( G, node_color="w", node_edgecolor="k", edge_color=ec,
                             edge_linewidth=2 )

    # Route
    orig = [9.72321, 52.3761]
    dest = [9.75413, 52.3554 ]
    length, route = trajectoryCalculation( G, orig, dest )

    print( length )
    printRoute( G, route )

#     print( segments )

    print( 'Finish.' )
