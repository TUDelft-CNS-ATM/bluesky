#!/usr/bin/python

"""

"""
from nommon.city_model.auxiliar import read_my_graphml
from nommon.city_model.city_graph import cityGraph
from nommon.city_model.corridors_implementation import corridorCreation, corridorLoad
from nommon.city_model.dynamic_segments import dynamicSegments
from nommon.city_model.multi_di_graph_3D import MultiDiGrpah3D
from nommon.city_model.no_fly_zones import restrictedSegments
from nommon.city_model.path_planning import trajectoryCalculation, printRoute
import os
import time

import configparser

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

    # No-fly zones
    # no_fly_coordinates = [[9.73, 52.39]]
    no_fly_coordinates = []
    restrictedSegments( G, segments, no_fly_coordinates, 0, 0, config )

    # Plotting the graph with corridors and no-fly zones
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
    dest = [9.766, 52.39 ]
    length, route = trajectoryCalculation( G, orig, dest )
    # Plotting the route
    print( length )
    printRoute( G, route )

#     print( segments )

    print( 'Finish.' )
