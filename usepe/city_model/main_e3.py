#!/usr/bin/python

"""
Main script which can be used to control all the other modules. This script can be adapted based
on the execution needs. However, the general steps that are contained in this master script are
explained below.

"""


import configparser
import pickle
import os
import time

from usepe.city_model.city_graph import cityGraph
from usepe.city_model.corridors_implementation import corridorLoad
from usepe.city_model.dynamic_segments import dynamicSegments
from usepe.city_model.multi_di_graph_3D import MultiDiGrpah3D
from usepe.city_model.no_fly_zones import restrictedSegments
from usepe.city_model.path_planning import trajectoryCalculation, printRoute
from usepe.city_model.scenario_definition import createFlightPlan, drawBuildings, \
    automaticFlightPlan
from usepe.city_model.strategic_deconfliction import deconflcitedScenario, initialPopulation
from usepe.city_model.utils import read_my_graphml, layersDict
import numpy as np
import osmnx as ox


if __name__ == '__main__':

    # -------------- 1. CONFIGURATION FILE -----------------
    """
    This section reads the configuration file.
    Change the config_path to read the desired file
    """
    # CONFIG
    config_path = "usepe/city_model/settings_e3.cfg"
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
        # fig, ax = ox.plot_graph( G )
    else:
        G = cityGraph( config )

    # -------------- 3. SEGMENTS ----------------------------
    """
    This section creates a airspace segmentation or loads the segmentation defined with the segment
    section of the configuration file.
    """
    if config['Segments'].getboolean( 'import' ):
        path = config['Segments']['path']
        with open(path, 'rb') as f:
            segments = pickle.load(f)
    else:
        segments = None

    # G, segments = dynamicSegments( G, config, segments, deleted_segments=None )

    # -------------- 4. CORRIDORS ---------------------------
    """
    This section loads the corridors defined with the corridor section of the configuration file
    Comment it to neglect the creation of corridors
    """
    G, segments = corridorLoad( G, segments, config )
    segments['873']['speed'] = 15
    segments['873']['capacity'] = 10
    G, segments = dynamicSegments( G, config, segments, deleted_segments=None )

    # -------------- 5. No-FLY ZONES ------------------------
    """
    This section adds restricted area (no-fly zones) by imposing zero velocity to the sectors that
    intersect with the area

    restricted area definition: it should be defined as a list of tuples (longitude, latitude) of
                                the vertices of the polygon defining the restricted area.
                                First and last point should be the same
    """
    step = False
    if step:
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
    orig = [9.72996, 52.44893]  # origin point
    dest = [9.71846, 52.45044]  # destination point
    travel_time, route = trajectoryCalculation( G, orig, dest )
    print( 'The travel time of the route is {0}'.format( travel_time ) )
    # print( 'The route is {0}'.format( route ) )
    # printRoute( G, route )

    # -------------- 7. Scenario definition -----------------------
    step1 = False
    step2 = False
    step3 = False
    """
    This section computes scenarios to be used in BlueSky.
    1. Path planning
    2. Automatic flight plan
    3. Print buildings
    """

    # The layer information will be used by the "createFlightPlan" function
    layers_dict = layersDict( config )

    if step1:
        """
        1.Path Planning
        We generate the flight plan of one drone. A scenario file is generated, which can be loaded by
        BlueSky. The "createFlightPlan" function transforms the optimal path (list of waypoints) to
        BlueSky commands
        """

        name = 'U001'
        ac = {'id': name, 'type': 'M600', 'accel': 3.5, 'v_max': 18, 
            'vs_max': 5, 'safety_volume_size': 1}
        departure_time = '00:00:00.00'
        scenario_path = r'usepe/city_model/scenario/scenario_corridor.scn'
        scenario_file = open( scenario_path, 'w' )
        createFlightPlan( route, ac, departure_time, G, layers_dict, scenario_file )
        scenario_file.close()

    if step2:
        """
        2. Automatic flight plan
        We generate many flight plans. Origins and destinations of a defined number of trajectories are
        randomly generated. We create a BlueSky scenario for each drone and a general BlueSky scenario
        which calls all the drone flight plans. Loading this general path in BlueSKy, we can simulate
        XX drones flying at the same time.
        """

        total_drones = 10
        base_name = 'U'
        scenario_general_path_base = r'usepe/city_model/scenario/scenario_10_drones'
        automaticFlightPlan( total_drones, base_name, G, layers_dict, scenario_general_path_base )

    if step3:
        """
        3. Draw buildings
        We generate the scenarios needed for printing the buildings in BlueSky.
        """
        time = '00:00:00.00'
        scenario_path_base = r'usepe/city_model/scenario/scenario_buildings_graph'
        drawBuildings( config, scenario_path_base, time )

    # -------------- 8. Strategic deconfliction -----------------------
    """
    This section computes a strategic deconflicted trajectory from origin to destination. An
    empty initial population is generated.
    """

    step = False
    sec_8_multi = True

    if step:
        orig = [9.72996, 52.44893 ]  # origin point
        dest = [9.71846, 52.45044]  # destination point
        name = 'U1'
        ac = {'id': name, 'type': 'M600', 'accel': 3.5, 'v_max': 18, 
            'vs_max': 5, 'safety_volume_size': 1}
        departure_time = 60  # seconds
        initial_time = 0  # seconds
        final_time = 1800  # seconds
        users = initialPopulation( segments, initial_time, final_time )
        scenario_path = r'scenario/USEPE/deconflicted_scenario.scn'
        scenario_file = open( scenario_path, 'w' )
        users = deconflcitedScenario( orig, dest, ac, departure_time, G, users, initial_time,
                                      final_time, segments, layers_dict, scenario_file, config )
        scenario_file.close()

    if sec_8_multi:
        op_plans = [
            {'ac': {'id': 'UAS1', 'type': 'M600', 'accel': 3.5, 'v_max': 18, 'vs_max': 5, 'safety_volume_size': 1}, 
                'base_coord': [9.72996, 52.44893], 'op_coord': [9.71846, 52.45044], 'departure_time': 0},
            {'ac': {'id': 'UAS2', 'type': 'M600', 'accel': 3.5, 'v_max': 18, 'vs_max': 5, 'safety_volume_size': 1}, 
                'base_coord': [9.69446, 52.43906], 'op_coord': [9.67161, 52.44252], 'departure_time': 0},
            {'ac': {'id': 'UAS3', 'type': 'M600', 'accel': 3.5, 'v_max': 18, 'vs_max': 5, 'safety_volume_size': 1}, 
                'base_coord': [9.65075, 52.42035], 'op_coord': [9.68234, 52.44066], 'departure_time': 0},
            {'ac': {'id': 'UAS4', 'type': 'M600', 'accel': 3.5, 'v_max': 18, 'vs_max': 5, 'safety_volume_size': 1}, 
                'base_coord': [9.65075, 52.42035], 'op_coord': [9.67879, 52.43126], 'departure_time': 30},
            {'ac': {'id': 'UAS5', 'type': 'M600', 'accel': 3.5, 'v_max': 18, 'vs_max': 5, 'safety_volume_size': 1}, 
                'base_coord': [9.68395, 52.43218], 'op_coord': [9.68334, 52.43572], 'departure_time': 285}
        ]
        initial_time = 0
        final_time = 1800
        users = initialPopulation(segments, initial_time, final_time)
        for plan in op_plans:
            scenario_path = r'scenario/USEPE/exercise_3/' + plan['ac']['id'] + r'_stage1.scn'
            scenario_file = open(scenario_path, 'w')
            users, route = deconflcitedScenario( plan['base_coord'], plan['op_coord'], plan['ac'], 
                                          plan['departure_time'], G, users, initial_time, 
                                          final_time, segments, layers_dict, scenario_file, config )
            scenario_file.close()

            scenario_path = r'scenario/USEPE/exercise_3/' + plan['ac']['id'] + r'_stage2.scn'
            scenario_file = open(scenario_path, 'w')
            users, route = deconflcitedScenario( plan['op_coord'], plan['base_coord'], plan['ac'], 
                                          plan['departure_time'] + 600, G, users, initial_time, 
                                          final_time, segments, layers_dict, scenario_file, config )
            scenario_file.close()

    # -----------------------------------------------------------------

    print( 'Finish.' )
