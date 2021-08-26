#!/usr/bin/python

"""

"""
import configparser
import time

from nommon.city_model.auxiliar import read_my_graphml
from nommon.city_model.city_graph import cityGraph
from nommon.city_model.corridors_implementation import corridorCreation
from nommon.city_model.dynamic_segments import dynamicSegments
from nommon.city_model.multi_di_graph_3D import MultiDiGrpah3D
from nommon.city_model.path_planning import trajectoryCalculation, printRoute
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

#     fig, ax = ox.plot_graph( G )
    # Segments
#     G, segments = dynamicSegments( G, config, segments=None )

#     segments[list( segments.keys() )[0]]['new'] = True
#     segments[list( segments.keys() )[3]]['updated'] = True
#     segments[list( segments.keys() )[3]]['speed'] = 0
#     G, segments = dynamicSegments( G, config, segments )

#     print( 'Saving the graph...' )
#     filepath = "./data/hannover_segments.graphml"
#     ox.save_graphml( G, filepath )
#     np.save( 'my_segments.npy', segments )

    # Corridors
#     G, segments = corridorCreation( G, segments, ( [9.73, 52.375], [9.77, 52.399 ] ),
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
    segments['segment_1_2_0']['speed'] = 0
    segments['segment_1_2_0']['updated'] = True
    segments['segment_1_2_1']['speed'] = 0
    segments['segment_1_2_1']['updated'] = True
    segments['segment_1_3_0']['speed'] = 0
    segments['segment_1_3_0']['updated'] = True
    segments['segment_1_3_1']['speed'] = 0
    segments['segment_1_3_1']['updated'] = True
#     segments['segment_2_3_0']['speed'] = 0
#     segments['segment_2_3_0']['updated'] = True
#     segments['segment_2_3_1']['speed'] = 0
#     segments['segment_2_3_1']['updated'] = True
#     segments['segment_3_2_0']['speed'] = 0
#     segments['segment_3_2_0']['updated'] = True
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
    ec = ["r" if G.edges[( u, v, k )]['speed'] == 0 else "gray" for u, v, k in G.edges( keys=True )]
    fig, ax = ox.plot_graph( G, node_color="w", node_edgecolor="k", edge_color=ec,
                             edge_linewidth=2 )

    # Route
    orig = [9.73, 52.38]
    dest = [9.77, 52.39 ]
    length, route = trajectoryCalculation( G, orig, dest )
    print( length )
    printRoute( G, route )

#     print( segments )

    print( 'Finish.' )
