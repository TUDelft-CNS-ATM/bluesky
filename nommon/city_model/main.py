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
        fig, ax = ox.plot_graph( G )
    else:
        G = cityGraph( config )

    # -------------- 3. SEGMENTS ----------------------------
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

    # -------------- 4. CORRIDORS ---------------------------
    """
    This section loads the corridors defined with the corridor section of the configuration file
    Comment it to neglect the creation of corridors
    """
    G, segments = corridorLoad( G, segments, config )

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
    """
    This section computes an optimal trajectory from origin to destination. The trajectory is
    optimal according to travel time.
    Comment it to no calculate an optimal trajectory
    Introduce origin and destination points inside the graph
    """
    orig = [9.77, 52.39 ]  # origin point
    dest = [9.73, 52.38]  # destination point
    travel_time, route = trajectoryCalculation( G, orig, dest )
    print( 'The travel time of the route is {0}'.format( travel_time ) )
    print( 'The route is {0}'.format( route ) )
    printRoute( G, route )

    # -------------- 7. Scenario definition -----------------------
    """
    This section computes scenarios to be used in BlueSky.
    1. Path planning
    2. Automatic flight plan
    3. Print buildings
    """

    # The layer information will be used by the "createFlightPlan" function
    layers_dict = layersDict( config )

    """
    1.Path Planning
    We generate the flight plan of one drone. A scenario file is generated, which can be loaded by
    BlueSky. The "createFlightPlan" function transforms the optimal path (list of waypoints) to
    BlueSky commands
    """

    ac = 'U001'
    departure_time = '00:00:00.00'
    scenario_path = r'.\data\scenario_example.scn'
    scenario_file = open( scenario_path, 'w' )
    createFlightPlan( route, ac, departure_time, G, layers_dict, scenario_file )
    scenario_file.close()

    """
    2. Automatic flight plan
    We generate many flight plans. Origins and destinations of a defined number of trajectories are
    randomly generated. We create a BlueSky scenario for each drone and a general BlueSky scenario
    which calls all the drone flight plans. Loading this general path in BlueSKy, we can simulate
    XX drones flying at the same time.
    """

    total_drones = 10
    base_name = 'U'
    scenario_general_path_base = r'.\data\scenario_10_drones'
    automaticFlightPlan( total_drones, base_name, G, layers_dict, scenario_general_path_base )

    """
    3. Draw buildings
    We generate the scenarios needed for printing the buildings in BlueSky.
    """
    time = '00:00:00.00'
    scenario_path_base = r'.\data\scenario_buildings_graph'
    drawBuildings( config, scenario_path_base, time )

    # ---------------------------------------------------------------------------------

    print( 'Finish.' )

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
#    # Plot graph
#     ec = ["r" if G.edges[( u, v, k )]['speed'] == 0 else "gray" for u, v, k in G.edges( keys=True )]
#     fig, ax = ox.plot_graph( G, node_color="w", node_edgecolor="k", edge_color=ec,
#                              edge_linewidth=2 )

