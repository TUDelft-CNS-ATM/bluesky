#!/usr/bin/python

"""
Main script which can be used to control all the other modules. This script can be adapted based
on the execution needs. However, the general steps that are contained in this master script are
explained below.

"""


import configparser
import os
import time

from nommon.city_model.city_graph import cityGraph
from nommon.city_model.corridors_implementation import corridorCreation, corridorLoad
from nommon.city_model.dynamic_segments import dynamicSegments
from nommon.city_model.multi_di_graph_3D import MultiDiGrpah3D
from nommon.city_model.no_fly_zones import restrictedSegments
from nommon.city_model.path_planning import trajectoryCalculation, printRoute
from nommon.city_model.scenario_definition import createFlightPlan, drawBuildings, \
    automaticFlightPlan
from nommon.city_model.utils import read_my_graphml, layersDict
import numpy as np
import osmnx as ox


__author__ = 'jbueno'
__copyright__ = '(c) Nommon 2021'


if __name__ == '__main__':

    # -------------- 1. CONFIGURATION FILE -----------------
    """
    This section reads the configuration file.
    Change the config_path to read the desired file
    """
    # CONFIG
    config_path = "C:/workspace3/bluesky/nommon/city_model/settings.cfg"
    config = configparser.ConfigParser()
    config.read( config_path )

    # -------------- 2. CITY GRAPH -------------------------
    """
    This section creates a city graph or loads the graph defined with the city section of the
    configuration file.
    """
    # City
    if config['City'].getboolean( 'import' ):
        filepath = config['City']['imported_graph_path']
        G = read_my_graphml( filepath )
        G = MultiDiGrpah3D( G )
#         fig, ax = ox.plot_graph( G )
        segments = np.load( 'my_segments.npy', allow_pickle='TRUE' ).item()
    else:
        G = cityGraph( config )

    # -------------- 3. CORRIDORS ---------------------------
    """
    This section loads the corridors defined with the corridor section of the configuration file
    Comment it to neglect the creation of corridors
    """
    G, segments = corridorLoad( G, segments, config )

    # -------------- 4. SEGMENTS ----------------------------
    """
    This section creates a airspace segmentation or loads the segmentation defined with the segment
    section of the configuration file.
    Comment it to neglect the segmentation
    """
    if config['Segments'].getboolean( 'import' ):
        path = config['Segments']['path']
        segments = np.load( path, allow_pickle='TRUE' ).item()
    else:
        segments = None

    G, segments = dynamicSegments( G, config, segments, deleted_segments=None )
    # -------------- 5. No-FLY ZONES ------------------------
    """
    This section adds restricted area (no-fly zones) by imposing zero verlocity to the sectors that
    intersect with the area

    restricted area definition: it should be defined as a list of tuples (longitude, latitude) of
                                the vertices of the polygon defining the restricted area.
                                First and last point should be the same
    """
    restricted_area = [( 9.78, 52.36 ),
                       ( 9.78, 52.365 ),
                       ( 9.77, 52.365 ),
                       ( 9.78, 52.36 )]
    G, segments = restrictedSegments( G, segments, restricted_area, config )

    # -------------- 6. PATH PLANNING -----------------------


#     fig, ax = ox.plot_graph( G )
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
#     G, segments = corridorCreation( G, segments, ( [9.73, 52.375], [9.77, 52.399 ] ),
#                                     200, 100, 50, 'COR_1' )
#     G, segments = dynamicSegments( G, config, segments=None )
    # No-fly zones
#     segments['segment_3_1_0']['speed'] = 0
#     segments['segment_3_1_0']['updated'] = True
#     segments['segment_3_1_1']['speed'] = 0
#     segments['segment_3_1_1']['updated'] = True
#     segments['segment_2_2_0']['speed'] = 0
#     segments['segment_2_2_0']['updated'] = True
#     segments['segment_2_2_1']['speed'] = 0
#     segments['segment_2_2_1']['updated'] = True
#     segments['segment_1_2_0']['speed'] = 0
#     segments['segment_1_2_0']['updated'] = True
#     segments['segment_1_2_1']['speed'] = 0
#     segments['segment_1_2_1']['updated'] = True
#     segments['segment_1_3_0']['speed'] = 0
#     segments['segment_1_3_0']['updated'] = True
#     segments['segment_1_3_1']['speed'] = 0
#     segments['segment_1_3_1']['updated'] = True
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
#     G, segments = dynamicSegments( G, config, segments )
#     edges = ox.utils_graph.graph_to_gdfs( G, nodes=False, fill_edge_geometry=False )

    # Plot graph
#     ec = ["r" if G.edges[( u, v, k )]['speed'] == 0 else "gray" for u, v, k in G.edges( keys=True )]
#     fig, ax = ox.plot_graph( G, node_color="w", node_edgecolor="k", edge_color=ec,
#                              edge_linewidth=2 )

    # Route
#     orig = [9.77, 52.39 ]
#     dest = [9.73, 52.38]
#     length, route = trajectoryCalculation( G, orig, dest )
#     print( 'The length of the route is {0}'.format( length ) )
#     print( 'The route is {0}'.format( route ) )
#     printRoute( G, route )

    # Path Planning
    layers_dict = layersDict( config )

#     ac = 'U005'
#     departure_time = '00:00:00.00'
#     scenario_path = r'C:\workspace3\bluesky\nommon\city_model\scenario5.scn'
#     scenario_file = open( scenario_path, 'w' )
#     createFlightPlan( route, ac, departure_time, G, layers_dict, scenario_file )
#     scenario_file.close()

    automaticFlightPlan( 10, 'U', G, layers_dict )

    # Drawing buildings
#     print( 'Creating scenario...' )
#     time = '00:00:00.00'
#     scenario_path_base = r'C:\workspace3\bluesky\nommon\city_model\scenario_buildings_graph'
#     drawBuildings( config, scenario_path_base, time )


#     G, segments = corridorLoad( G, segments, config )
#
#     # No-fly zones
#     # no_fly_coordinates = [[9.73, 52.39]]
#     no_fly_coordinates = []
#     restrictedSegments( G, segments, no_fly_coordinates, 0, 0, config )
#
#     # Plotting the graph with corridors and no-fly zones
#     ec = []
#     for u, v, k in G.edges( keys=True ):
#         if G.edges[( u, v, k )]['speed'] == 0:
#             ec.append( "r" )
#         elif G.edges[( u, v, k )]['speed'] == 100:
#             ec.append( "g" )
#         else:
#             ec.append( "gray" )
#     fig, ax = ox.plot_graph( G, node_color="w", node_edgecolor="k", edge_color=ec,
#                              edge_linewidth=2 )
#
#     # Route
#     orig = [9.72321, 52.3761]
#     dest = [9.766, 52.39 ]
#     length, route = trajectoryCalculation( G, orig, dest )
#     # Plotting the route
#     print( length )
#     printRoute( G, route )


    print( 'Finish.' )
