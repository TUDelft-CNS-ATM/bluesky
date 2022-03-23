#!/usr/bin/python

"""

"""
import configparser
import copy
import datetime
import json
import math
import os
import random
import string
import sys

from pyproj import Transformer

from usepe.city_model.building_height import readCity
from usepe.city_model.multi_di_graph_3D import MultiDiGrpah3D
from usepe.city_model.path_planning import trajectoryCalculation
from usepe.city_model.utils import read_my_graphml, checkIfNoFlyZone, layersDict
import osmnx as ox
import pandas as pd


# from usepe.city_model.strategic_deconfliction import initialPopulation, deconflcitedScenario
__author__ = 'jbueno'
__copyright__ = '(c) Nommon 2021'


def headingIncrement( actual_heading, wpt1, wpt2, G ):
    """
    Compute the heading from waypoint 1 (wpt1) to waypoint 2 (wpt2). In addition, it returns the
    heading increment between the new heading and the actual heading.

    Args:
            actual_heading (float): drone heading
            wpt1 (string): string indicating the waypoint the drone is heading to
            wpt2 (string) string indicating the next waypoint
            G (graph)

    Returns:
            new_heading (float): float indicating the new heading
            increment (float): float indicating the difference between the new heading and the
            actual heading
    """
    y = G.nodes[wpt2]['y'] - G.nodes[wpt1]['y']
    x = G.nodes[wpt2]['x'] - G.nodes[wpt1]['x']
    theta = -math.atan2( y, x ) * 180 / math.pi
    if theta < 0:
        theta += 360

    new_heading = theta + 90
    if new_heading >= 360:
        new_heading = new_heading - 360
    increment = abs( new_heading - actual_heading )
    return new_heading, increment


def calcDistAccel( speed_t0, speed_tf, ac ):
    """
    Compute the distance used to decelerate or accelerate

    Args:
            speed_t0 (float): initial speed [m/s]
            speed_tf (float): final speed [m/s]
            ac (dictionary): aircraft parameters {id, type, accel, v_max, vs_max}

    Returns:
            dist (float): distance [m]
            time (float): time needed to decelerate or accelerate [s]
    """
    time = abs( speed_tf - speed_t0 ) / ac['accel']
    if speed_t0 > speed_tf:
        dist = speed_t0 * time - ac['accel'] * time * time / 2
    else:
        dist = speed_t0 * time + ac['accel'] * time * time / 2

    return dist, time


def turnDefinition( increment, ac, speed, climbing=None ):
    """
    Compute some parameters needed to control the change of direction of the drone. It returns the
    turn speed, the turn distance and turn radius.

    Args:
            increment (float): float indicating the change of heading
            ac (dictionary): aircraft parameters {id, type, accel, v_max, vs_max}
            speed (float): drone's speed before the turn [m/s]

    Returns:
            turn_speed (float or None): indicating the velocity when performing the turn
            turn_dist (float or None): indicating the distance at which the drone has to start to
            decelerate [nm]
            turn_rad (float or None): indicating the turn radius
    """
    m2nm = 0.000539957
    m_s2knot = 1.944  # m/s to knots

    if increment < 20:
        turn_speed = None
        turn_dist = None
        turn_rad = None
    elif increment < 45:
        turn_speed = 20
        dist, time = calcDistAccel( speed, turn_speed / m_s2knot, ac )
        turn_dist = dist * m2nm
        turn_rad = 0.0003
    elif increment < 60:
        turn_speed = 17
        dist, time = calcDistAccel( speed, turn_speed / m_s2knot, ac )
        turn_dist = dist * m2nm
        turn_rad = 0.0003
    elif increment < 70:
        turn_speed = 15
        dist, time = calcDistAccel( speed, turn_speed / m_s2knot, ac )
        turn_dist = dist * m2nm
        turn_rad = 0.0003
    elif increment < 80:
        turn_speed = 10
        dist, time = calcDistAccel( speed, turn_speed / m_s2knot, ac )
        turn_dist = dist * m2nm
        turn_rad = 0.0003
    elif increment < 90:
        turn_speed = 8
        dist, time = calcDistAccel( speed, turn_speed / m_s2knot, ac )
        turn_dist = dist * m2nm
        turn_rad = 0.0002
    elif climbing:
        turn_speed = 1
        dist, time = calcDistAccel( speed, turn_speed / m_s2knot, ac )
        turn_dist = dist * m2nm
        turn_rad = 0.0001
    else:
        turn_speed = 5
        dist, time = calcDistAccel( speed, turn_speed / m_s2knot, ac )
        turn_dist = dist * m2nm
        turn_rad = 0.0001

    return turn_speed, turn_dist, turn_rad


# # DEPRECIATED
# def turnDetection( route, i, dist, G, scenario_commands ):
#     m2nm = 0.000539957
#     dist_step = 0
#     j = i - 1
#     while ( j >= 0 ) & ( dist_step < dist ):
#         dist_step = ox.distance.great_circle_vec( G.nodes[route[i]]['y'], G.nodes[route[i]]['x'],
#                                                   G.nodes[route[j]]['y'], G.nodes[route[j]]['x'] )
#         dist_step = dist_step * m2nm
#         if dist_step < dist:
#             scenario_commands[route[j]]['additional'] = ''
#
#         j -= 1
#
#     return scenario_commands


def turnDetectionV2( route_parameters, i ):
    """
    The condition of deceleration may overlap between two waypoints. This function update the turning
    information to take the most restrictive condition. In addition, it returns a variable indicating
    which command has to be written.

    Args:
            route_parameters (dictionary): dictionary with all the information about the route
            i (integer): integer indicating the node of the route list

    Returns:
            option (integer): variable indicating which commands have to be imposed
    """
    total_dist = 0
    maximum_turn_distance = 0.05
    m2nm = 0.000539957
    j = i + 1
    option = 1
    while ( total_dist < maximum_turn_distance ) & ( str( j ) in route_parameters ):
#         total_dist += route_parameters[str( j - 1 )]['dist'] * m2nm
        total_dist = ox.distance.great_circle_vec( route_parameters[str( i )]['lat'],
                                                   route_parameters[str( i )]['lon'],
                                                   route_parameters[str( j )]['lat'],
                                                   route_parameters[str( j )]['lon'] )
        total_dist = total_dist * m2nm
        if route_parameters[str( j )]['turn dist'] is None:  # 4)
            pass
        elif total_dist > route_parameters[str( j )]['turn dist']:  # 3)
            pass
        elif route_parameters[str( i )]['turn dist'] + total_dist > \
            route_parameters[str( j )]['turn dist']:  # 1)

            if route_parameters[str( i )]['turn speed'] <= route_parameters[str( j )]['turn speed']:  # 1 A)
                route_parameters[str( j )]['turn dist'] = total_dist * m2nm - 0.000001

            else:  # 1 B)
                if option != 3:
                    option = 2

        else:  # 2)
            option = 3

        j += 1

    return option


def routeParameters( G, route, ac ):
    """
    Compute all the information about the route (e.g. turn distance, turn speed, altitude, etc.). It
    is stored as a dictionary.

    Args:
            G (grah)
            route (list): list of waypoints
            ac (dictionary): aircraft parameters {id, type, accel, v_max, vs_max}

    Returns:
            route_parameters (dictionary): dictionary with all the information about the route
    """

    points_to_check_speed_reduction = 5
    speed_reduction = len( route ) > points_to_check_speed_reduction
    distance_speed_reduction = 125  # m

    route_parameters = {}
    final_node = {}
    for i in range( len( route ) - 1 ):
        node = {}
        name = route[i]
        if i == 0:
            new_heading, increment = headingIncrement( 0, name, route[i + 1], G )
            node['name'] = name
            node['lat'] = G.nodes[name]['y']
            node['lon'] = G.nodes[name]['x']
            node['alt'] = G.nodes[name]['z']
            node['turn speed'] = None
            node['turn rad'] = None
            node['turn dist'] = None
            node['hdg'] = new_heading
            if name[0] == route[i + 1][0]:
                node['speed'] = max( min( G.edges[( name, route[i + 1], 0 )]['speed'], ac['v_max'] ), 0.001 )
            else:
                node['speed'] = max( min( G.edges[( name, route[i + 1], 0 )]['speed'], ac['vs_max'] ), 0.001 )
            node['dist'] = G.edges[( name, route[i + 1], 0 )]['length']
        else:
            new_heading, increment = headingIncrement( route_parameters[str( i - 1 )]['hdg'], name,
                                                       route[i + 1], G )
            if name[0] == route[i + 1][0]:
                turn_speed, turn_dist, turn_rad = turnDefinition( increment, ac,
                                                                  route_parameters[str( i - 1 )]['speed'] )
            else:
                turn_speed, turn_dist, turn_rad = turnDefinition( increment, ac,
                                                                  route_parameters[str( i - 1 )]['speed'],
                                                                  climbing=True )

            if turn_speed:
                if turn_speed > route_parameters[str( i - 1 )]['speed'] * 1.944:
                    turn_speed = None
                    turn_dist = None
                    turn_rad = None
            if i == 1:
                if turn_dist:
                    turn_dist = ( G.edges[( route[i - 1], name, 0 )]['length'] ) * 0.000539957 - 0.0001
            node['name'] = name
            node['lat'] = G.nodes[name]['y']
            node['lon'] = G.nodes[name]['x']
            node['alt'] = G.nodes[name]['z']
            node['turn speed'] = turn_speed
            node['turn rad'] = turn_rad
            node['turn dist'] = turn_dist
            node['hdg'] = new_heading
            if name[0] == route[i + 1][0]:
                node['speed'] = max( min( G.edges[( name, route[i + 1], 0 )]['speed'], ac['v_max'] ), 0.001 )
            else:
                node['speed'] = max( min( G.edges[( name, route[i + 1], 0 )]['speed'], ac['vs_max'] ), 0.001 )
            node['dist'] = G.edges[( name, route[i + 1], 0 )]['length']

            # Apply speed limitation for last nodes when too close to the end
            if speed_reduction:
                if i > ( len( route ) - points_to_check_speed_reduction ):
                    distance = ox.distance.great_circle_vec( G.nodes[route[i]]['x'],
                                                             G.nodes[route[i]]['y'],
                                                             G.nodes[route[i + 1]]['x'],
                                                             G.nodes[route[i + 1]]['y'] )
                    if distance < distance_speed_reduction or i == len( route ) - 2:
                        node['speed'] = 5  # m/s
                        final_node['speed'] = 5  # m/s
                    else:
                        final_node['speed'] = max( min( G.edges[( name, route[i + 1], 0 )]['speed'], ac['vs_max'] ), 0.001 )
            else:
                final_node['speed'] = max( min( G.edges[( name, route[i + 1], 0 )]['speed'], ac['vs_max'] ), 0.001 )

        route_parameters[str( i )] = node

    final_node['name'] = route[-1]
    final_node['lat'] = G.nodes[route[-1]]['y']
    final_node['lon'] = G.nodes[route[-1]]['x']
    final_node['alt'] = G.nodes[route[-1]]['z']
    final_node['turn speed'] = None
    final_node['turn rad'] = None
    final_node['turn dist'] = None
    final_node['hdg'] = None
    # final_node['speed'] = None
    final_node['dist'] = None

    route_parameters[str( len( route ) - 1 )] = final_node

    return route_parameters


def createInstructionV3( scenario_file, route_parameters, i, ac, G, layers_dict, time, state ):
    """
    Write the commands associated to the waypoint i in the scenario file. It also returns the
    variable "state" with information about the state of the drone once it has done the commands for
    the waypoint i.

    Args:
            scenario_file: text file where we write the commands
            route_parameters (dictionary): dictionary with all the information about the route
            i (integer): integer indicating the node of the route list
            ac (dictionary): aircraft parameters {id, type, accel, v_max, vs_max}
            G (graph)
            layers_dict (dictionary): dictionary with the information about layers and altitudes
            time (string): time for the instructions
            state (dictionary): information about the state of the drone before doing the
            commands for the waypoint i

    Returns:
            state (dictionary): information about the state of the drone once it has done the
            commands for the waypoint i
    """
    wpt1 = route_parameters[str( i )]['name']
    wpt2 = route_parameters[str( i + 1 )]['name']
    m_s2knot = 1.944  # m/s to knots
    m2ft = 3.281  # m to ft
    m_s2ft_min = 197  # m/s to ft/min

    if state['action'] is None:  # if it is the first waypoint

        if wpt1[0] == wpt2[0]:  # wpt1 and wpt2 in the same layer
            hdg = route_parameters[str( i )]['hdg']
            new_line1 = '{0} > CRE {1} {7} {2} {3} {4} {5} {6}'.format( 
                time, ac['id'], route_parameters[str( i )]['lat'], route_parameters[str( i )]['lon'],
                hdg, route_parameters[str( i )]['alt'] * m2ft,
                0, ac['type'] )

            new_line2 = '{0} > SPD {1} {2}'.format( 
                time, ac['id'], str( route_parameters[str( i )]['speed'] * m_s2knot ) )

            state['action'] = 'cruise'
            state['heading'] = hdg

            scenario_file.write( new_line1 + '\n' + new_line2 + '\n' )
        else:
            new_line0 = '{0} > DEFWPT {1},{2},{3}'.format( 
                time, wpt1, route_parameters[str( i )]['lat'], route_parameters[str( i )]['lon'] )
            new_line1 = '{0} > CRE {1} {4} {2} 0 {3} 0.1'.format( 
                time, ac['id'], wpt1, route_parameters[str( i )]['alt'] * m2ft, ac['type'] )
            new_line2 = '{0} > ADDWPT {1} {2}, , {3}'.format( 
                time, ac['id'], wpt1, str( route_parameters[str( i )]['speed'] * m_s2knot ) )
            new_line3 = '{0} > ALT {1} {2}'.format( time, ac['id'],
                                                    route_parameters[str( i + 1 )]['alt'] * m2ft )
            new_line4 = '{0} > VS {1} {2}'.format( 
                time, ac['id'], str( route_parameters[str( i )]['speed'] * m_s2ft_min ) )
            state['ref_wpt'] = wpt1
            state['action'] = 'climbing'
            state['heading'] = 0

            scenario_file.write( new_line0 + '\n' + new_line1 + '\n' + new_line2 + '\n' +
                                 new_line3 + '\n' + new_line4 + '\n' )

        return state

    if wpt1[0:3] == 'COR':  # in a corridor
        """
        Steps of the corridor:

        if state['action'] == 'entering_corridor' --> wpt1 = COR_XXX_in_1 and wpt2 = COR_XXX_in_2 -->
            state['action'] == 'climbing_corridor'

        if state['action'] == 'climbing_corridor' --> wpt1 = COR_XXX_in_2 and wpt2 = COR_XXX -->
            state['action'] == 'corridor'

        if state['action'] == 'corridor' --> wpt1 = COR_XXX and wpt2 = COR_XXX -->
            state['action'] == 'corridor'        (n times)

        if state['action'] == 'corridor' --> wpt1 = COR_XXX and wpt2 = COR_XXX_in_2 -->
            state['action'] == 'descending_corridor'

        if state['action'] == 'descending_corridor' --> wpt1 = COR_XXX_in_2 and wpt2 = COR_XXX_in_1 -->
            state['action'] == 'leaving_corridor'

        if state['action'] == 'leaving_corridor' --> wpt1 = COR_XXX_in_1 and wpt2 = YXXX -->
            state['action'] == 'cruise'
        """
        if state['action'] == 'entering_corridor' or state['action'] == 'descending_corridor':
            new_line0 = '{0} > DEFWPT {1},{2},{3}'.format( 
                time, wpt1, route_parameters[str( i )]['lat'], route_parameters[str( i )]['lon'] )
            new_line1 = '{0} > ADDWPT {1} FLYOVER'.format( time, ac['id'] )
            new_line2 = '{0} > ADDWPT {1} {2}, ,{3}'.format( 
                time, ac['id'], wpt1, str( route_parameters[str( i )]['speed'] * m_s2knot ) )
            new_line3 = '{0} > {1} ATDIST {2} {3} SPD {4} {5}'.format( 
                time, ac['id'], wpt1, 0.03, ac['id'], 5 )
            new_line4 = '{0} > {1} AT {2} DO {3} SPD 0'.format( 
                time, ac['id'], wpt1, ac['id'] )
            new_line5 = '{0} > {1} AT {2} DO {3} ALT {4}'.format( 
                time, ac['id'], wpt1, ac['id'], route_parameters[str( i + 1 )]['alt'] * m2ft )
            new_line6 = '{0} > {1} AT {2} DO {3} VS {4}'.format( 
                time, ac['id'], wpt1, ac['id'], str( route_parameters[str( i )]['speed'] * m_s2ft_min ) )

            scenario_file.write( new_line0 + '\n' + new_line1 + '\n' + new_line2 + '\n' +
                                 new_line3 + '\n' + new_line4 + '\n' + new_line5 + '\n' +
                                 new_line6 + '\n' )
            state['ref_wpt'] = wpt1
            if wpt1[-4:] == 'in_2' and wpt2[-4:] == 'in_1':
                state['action'] = 'leaving_corridor'
            else:
                state['action'] = 'climbing_corridor'
        elif state['action'] == 'climbing_corridor' or state['action'] == 'leaving_corridor':
            new_line1 = '{0} > {1} AT {2} DO {3} ATALT {4}, LNAV {5} ON'.format( 
                time, ac['id'], state['ref_wpt'], ac['id'], route_parameters[str( i )]['alt'] * m2ft, ac['id'] )
            new_line2 = '{0} > {1} AT {2} DO {3} ATALT {4}, VNAV {5} ON'.format( 
                time, ac['id'], state['ref_wpt'], ac['id'], route_parameters[str( i )]['alt'] * m2ft, ac['id'] )
            new_line3 = '{0} > {1} AT {2} DO {3} ATALT {4}, ADDWPT {5} {6} {7}, {8}, {9}, {10}'.format( 
                time, ac['id'], state['ref_wpt'], ac['id'], route_parameters[str( i )]['alt'] * m2ft, ac['id'],
                route_parameters[str( i )]['lat'], route_parameters[str( i )]['lon'],
                route_parameters[str( i )]['alt'] * m2ft,
                str( route_parameters[str( i )]['speed'] * m_s2knot ), state['ref_wpt'] )

            scenario_file.write( new_line1 + '\n' + new_line2 + '\n' +
                                 new_line3 + '\n' )
            if wpt1[-4:] == 'in_1' and wpt2[-4:] != 'in_2':
                state['action'] = 'cruise'
            else:
                state['action'] = 'corridor'
            state['heading'] = route_parameters[str( i )]['hdg']
        elif state['action'] == 'corridor':
            new_line1 = '{0} > ADDWPT {1} FLYTURN'.format( time, ac['id'] )
            new_line2 = '{0} > ADDWPT {1} TURNSPD {2}'.format( time, ac['id'], 15 )
            new_line3 = '{0} > ADDWPT {1} {2} {3} {4} {5}'.format( 
                time, ac['id'], route_parameters[str( i )]['lat'], route_parameters[str( i )]['lon'],
                route_parameters[str( i )]['alt'] * m2ft,
                str( route_parameters[str( i )]['speed'] * m_s2knot ) )

            scenario_file.write( new_line1 + '\n' + new_line2 + '\n' + new_line3 + '\n' )

            if wpt2[-4:] == 'in_2':
                state['action'] = 'descending_corridor'
            else:
                state['action'] = 'corridor'

        return state

    if wpt1[0] == wpt2[0] or wpt2[0:3] == 'COR':  # wpt1 and wpt2 in the same layer or it is going to enter in a corridor
        if state['action'] == 'cruise':  # drone was flying horizontally
            turn_speed = route_parameters[str( i )]['turn speed']
            turn_dist = route_parameters[str( i )]['turn dist']
            turn_rad = route_parameters[str( i )]['turn rad']
            if turn_speed is None:
                new_line1 = '{0} > ADDWPT {1} FLYOVER'.format( time, ac['id'] )
                new_line2 = '{0} > ADDWPT {1} {2} {3}, , {4}'.format( 
                    time, ac['id'], route_parameters[str( i )]['lat'], route_parameters[str( i )]['lon'],
                    str( route_parameters[str( i )]['speed'] * m_s2knot ) )

                scenario_file.write( new_line1 + '\n' + new_line2 + '\n' )

            else:
                option = turnDetectionV2( route_parameters, i )
                if option == 1:
                    new_line0 = '{0} > DEFWPT {1},{2},{3}'.format( 
                        time, wpt1, route_parameters[str( i )]['lat'], route_parameters[str( i )]['lon'] )
                    new_line1 = '{0} > ADDWPT {1} FLYTURN'.format( time, ac['id'] )
                    new_line2 = '{0} > ADDWPT {1} TURNSPD {2}'.format( time, ac['id'], turn_speed )
                    new_line3 = '{0} > ADDWPT {1} TURNRAD {2}'.format( time, ac['id'], turn_rad )
                    new_line4 = '{0} > ADDWPT {1} {2}, , {3}'.format( 
                        time, ac['id'], wpt1, str( route_parameters[str( i )]['speed'] * m_s2knot ) )
                    new_line5 = '{0} > {1} ATDIST {2} {3} SPD {4} {5}'.format( 
                        time, ac['id'], wpt1, turn_dist, ac['id'], turn_speed )
                    new_line6 = '{0} > {1} AT {2} DO {3} LNAV ON'.format( 
                        time, ac['id'], wpt1, ac['id'] )
                    new_line7 = '{0} > {1} AT {2} DO {3} VNAV ON'.format( 
                        time, ac['id'], wpt1, ac['id'] )

                    scenario_file.write( new_line0 + '\n' + new_line1 + '\n' + new_line2 + '\n' +
                                         new_line3 + '\n' + new_line4 + '\n' + new_line5 + '\n' +
                                         new_line6 + '\n' + new_line7 + '\n' )
                elif option == 2:
                    new_line0 = '{0} > DEFWPT {1},{2},{3}'.format( 
                        time, wpt1, route_parameters[str( i )]['lat'], route_parameters[str( i )]['lon'] )
                    new_line1 = '{0} > ADDWPT {1} FLYTURN'.format( time, ac['id'] )
                    new_line2 = '{0} > ADDWPT {1} TURNSPD {2}'.format( time, ac['id'], turn_speed )
                    new_line3 = '{0} > ADDWPT {1} TURNRAD {2}'.format( time, ac['id'], turn_rad )
                    new_line4 = '{0} > ADDWPT {1} {2}, , {3}'.format( 
                        time, ac['id'], wpt1, str( route_parameters[str( i )]['speed'] * m_s2knot ) )
                    new_line5 = '{0} > {1} ATDIST {2} {3} SPD {4} {5}'.format( 
                        time, ac['id'], wpt1, turn_dist, ac['id'], turn_speed )

                    scenario_file.write( new_line0 + '\n' + new_line1 + '\n' + new_line2 + '\n' +
                                         new_line3 + '\n' + new_line4 + '\n' + new_line5 + '\n' )
                elif option == 3:
                    new_line1 = '{0} > ADDWPT {1} FLYTURN'.format( time, ac['id'] )
                    new_line2 = '{0} > ADDWPT {1} TURNSPD {2}'.format( time, ac['id'], turn_speed )
                    new_line3 = '{0} > ADDWPT {1} TURNRAD {2}'.format( time, ac['id'], turn_rad )
                    new_line4 = '{0} > ADDWPT {1} {2} {3}, , {4}'.format( 
                        time, ac['id'], route_parameters[str( i )]['lat'],
                        route_parameters[str( i )]['lon'],
                        str( route_parameters[str( i )]['speed'] * m_s2knot ) )

                    scenario_file.write( new_line1 + '\n' + new_line2 + '\n' +
                                         new_line3 + '\n' + new_line4 + '\n' )

        elif state['action'] == 'climbing':  # drone was climbing
            new_line1 = '{0} > {1} AT {2} DO {3} ATALT {4}, LNAV {5} ON'.format( 
                time, ac['id'], state['ref_wpt'], ac['id'], route_parameters[str( i )]['alt'] * m2ft, ac['id'] )
            new_line2 = '{0} > {1} AT {2} DO {3} ATALT {4}, VNAV {5} ON'.format( 
                time, ac['id'], state['ref_wpt'], ac['id'], route_parameters[str( i )]['alt'] * m2ft, ac['id'] )
            new_line3 = '{0} > {1} AT {2} DO {3} ATALT {4}, ADDWPT {5} {6} {7}, {8}, {9}, {10}'.format( 
                time, ac['id'], state['ref_wpt'], ac['id'], route_parameters[str( i )]['alt'] * m2ft, ac['id'],
                route_parameters[str( i )]['lat'], route_parameters[str( i )]['lon'],
                route_parameters[str( i )]['alt'] * m2ft,
                str( route_parameters[str( i )]['speed'] * m_s2knot ), state['ref_wpt'] )

            scenario_file.write( new_line1 + '\n' + new_line2 + '\n' +
                                 new_line3 + '\n' )

        if wpt2[0:3] == 'COR':
            state['action'] = 'entering_corridor'
        else:
            state['action'] = 'cruise'
        state['heading'] = route_parameters[str( i )]['hdg']
        return state

    elif wpt1[0] != wpt2[0]:  # wpt1 and wpt2 at different altitude
        new_line0 = '{0} > DEFWPT {1},{2},{3}'.format( 
            time, wpt1, route_parameters[str( i )]['lat'], route_parameters[str( i )]['lon'] )
        if state['action'] == 'cruise':  # drone was flying horizontally
            new_line1 = '{0} > ADDWPT {1} FLYOVER'.format( time, ac['id'] )
            new_line2 = '{0} > ADDWPT {1} {2}, ,{3}'.format( 
                time, ac['id'], wpt1, str( route_parameters[str( i )]['speed'] * m_s2knot ) )
            new_line3 = '{0} > {1} ATDIST {2} {3} SPD {4} {5}'.format( 
                time, ac['id'], wpt1, 0.015, ac['id'], 5 )
            new_line4 = '{0} > {1} AT {2} DO {3} SPD 0'.format( 
                time, ac['id'], wpt1, ac['id'] )
            new_line5 = '{0} > {1} AT {2} DO {3} ALT {4}'.format( 
                time, ac['id'], wpt1, ac['id'], route_parameters[str( i + 1 )]['alt'] * m2ft )
            new_line6 = '{0} > {1} AT {2} DO {3} VS {4}'.format( 
                time, ac['id'], wpt1, ac['id'], str( route_parameters[str( i )]['speed'] * m_s2ft_min ) )

            scenario_file.write( new_line0 + '\n' + new_line1 + '\n' + new_line2 + '\n' +
                                 new_line3 + '\n' + new_line4 + '\n' + new_line5 + '\n' +
                                 new_line6 + '\n' )

            state['ref_wpt'] = wpt1

        elif state['action'] == 'climbing':  # drone was climbing
            new_line1 = '{0} > {1} AT {2} DO {3} ATALT {4}, ALT {5} {6}'.format( 
                time, ac['id'], state['ref_wpt'], ac['id'], route_parameters[str( i )]['alt'] * m2ft, ac['id'],
                route_parameters[str( i + 1 )]['alt'] * m2ft )
            new_line2 = '{0} > {1} AT {2} DO {3} ATALT {4}, VS {5} {6}'.format( 
                time, ac['id'], state['ref_wpt'], ac['id'], route_parameters[str( i )]['alt'] * m2ft, ac['id'],
                str( route_parameters[str( i )]['speed'] * m_s2ft_min ) )

            scenario_file.write( new_line1 + '\n' + new_line2 + '\n' )

        state['action'] = 'climbing'
        state['heading'] = 0
        return state

# DEPRECIATED
# def createInstructionV2( scenario_file, route_parameters, i, ac, G, layers_dict, time, state ):
#     wpt1 = route_parameters[str( i )]['name']
#     wpt2 = route_parameters[str( i + 1 )]['name']
#     m_s2knot = 1.944
#     m2ft = 3.281
#     m_s2ft_min = 197
#
#     if state['action'] is None:
#         new_line0 = '{0} > DEFWPT {1},{2},{3}'.format(
#             time, wpt1, route_parameters[str( i )]['lat'], route_parameters[str( i )]['lon'] )
#
#         if wpt1[0] == wpt2[0]:
#             hdg = route_parameters[str( i )]['hdg']
#             new_line1 = '{0} > CRE {1} M600 {2} {3} {4} {5}'.format(
#                 time, ac, wpt1, hdg, route_parameters[str( i )]['alt'] * m2ft,
#                 str( G.edges[( wpt1, wpt2, 0 )]['speed'] * m_s2knot ) )
#
#             state['action'] = 'cruise'
#             state['heading'] = hdg
#
#             scenario_file.write( new_line0 + '\n' + new_line1 + '\n' )
#         else:
#             new_line1 = '{0} > CRE {1} M600 {2} 0 {3} 0'.format(
#                 time, ac, wpt1, route_parameters[str( i )]['alt'] * m2ft )
#             new_line2 = '{0} > ADDWPT {1} {2}, , {3}'.format(
#                 time, ac, wpt1, str( route_parameters[str( i )]['speed'] * m_s2knot ) )
#             new_line3 = '{0} > ALT {1} {2}'.format( time, ac,
#                                                     route_parameters[str( i + 1 )]['alt'] * m2ft )
#             new_line4 = '{0} > VS {1} {2}'.format(
#                 time, ac, str( route_parameters[str( i )]['speed'] * m_s2ft_min ) )
#             state['ref_wpt'] = wpt1
#             state['action'] = 'climbing'
#             state['heading'] = 0
#
#             scenario_file.write( new_line0 + '\n' + new_line1 + '\n' + new_line2 + '\n' +
#                                  new_line3 + '\n' + new_line4 + '\n' )
#
#         return state
#
#     if wpt1[0] == wpt2[0]:
#         new_line0 = '{0} > DEFWPT {1},{2},{3}'.format(
#             time, wpt1, route_parameters[str( i )]['lat'], route_parameters[str( i )]['lon'] )
#         if state['action'] == 'cruise':
#             turn_speed = route_parameters[str( i )]['turn speed']
#             turn_dist = route_parameters[str( i )]['turn dist']
#             turn_rad = route_parameters[str( i )]['turn rad']
#             if turn_speed is None:
#                 new_line1 = '{0} > ADDWPT {1} FLYOVER'.format( time, ac )
#                 new_line2 = '{0} > ADDWPT {1} {2}, , {3}'.format(
#                     time, ac, wpt1, str( route_parameters[str( i )]['speed'] * m_s2knot ) )
#
#                 scenario_file.write( new_line0 + '\n' + new_line1 + '\n' + new_line2 + '\n' )
#
#             else:
#                 new_line1 = '{0} > ADDWPT {1} FLYTURN'.format( time, ac )
#                 new_line2 = '{0} > ADDWPT {1} TURNSPD {2}'.format( time, ac, turn_speed )
#                 new_line3 = '{0} > ADDWPT {1} TURNRAD {2}'.format( time, ac, turn_rad )
#                 new_line4 = '{0} > ADDWPT {1} {2}, , {3}'.format(
#                     time, ac, wpt1, str( route_parameters[str( i )]['speed'] * m_s2knot ) )
#                 option = turnDetectionV2( route_parameters, i )
#                 if option == 1:
#                     new_line5 = '{0} > {1} ATDIST {2} {3} SPD {4} {5}'.format(
#                         time, ac, wpt1, turn_dist, ac, turn_speed )
#                     new_line6 = '{0} > {1} AT {2} DO LNAV {3} ON'.format(
#                         time, ac, wpt1, ac )
#                     new_line7 = '{0} > {1} AT {2} DO VNAV {3} ON'.format(
#                         time, ac, wpt1, ac )
#
#                     scenario_file.write( new_line0 + '\n' + new_line1 + '\n' + new_line2 + '\n' +
#                                          new_line3 + '\n' + new_line4 + '\n' + new_line5 + '\n' +
#                                          new_line6 + '\n' + new_line7 + '\n' )
#                 elif option == 2:
#                     new_line5 = '{0} > {1} ATDIST {2} {3} SPD {4} {5}'.format(
#                         time, ac, wpt1, turn_dist, ac, turn_speed )
#
#                     scenario_file.write( new_line0 + '\n' + new_line1 + '\n' + new_line2 + '\n' +
#                                          new_line3 + '\n' + new_line4 + '\n' + new_line5 + '\n' )
#                 elif option == 3:
#
#                     scenario_file.write( new_line0 + '\n' + new_line1 + '\n' + new_line2 + '\n' +
#                                          new_line3 + '\n' + new_line4 + '\n' )
#
#         elif state['action'] == 'climbing':
#             new_line1 = '{0} > {1} AT {2} DO {3} ATALT {4}, LNAV {5} ON'.format(
#                 time, ac, state['ref_wpt'], ac, route_parameters[str( i )]['alt'] * m2ft, ac )
#             new_line2 = '{0} > {1} AT {2} DO {3} ATALT {4}, VNAV {5} ON'.format(
#                 time, ac, state['ref_wpt'], ac, route_parameters[str( i )]['alt'] * m2ft, ac )
#             new_line3 = '{0} > {1} AT {2} DO {3} ATALT {4}, ADDWPT {5} {6}, {7}, {8}, {9}'.format(
#                 time, ac, state['ref_wpt'], ac, route_parameters[str( i )]['alt'] * m2ft, ac, wpt1,
#                 route_parameters[str( i )]['alt'] * m2ft,
#                 str( route_parameters[str( i )]['speed'] * m_s2knot ), state['ref_wpt'] )
#
#             scenario_file.write( new_line0 + '\n' + new_line1 + '\n' + new_line2 + '\n' +
#                                  new_line3 + '\n' )
#
#         state['action'] = 'cruise'
#         state['heading'] = route_parameters[str( i )]['hdg']
#         return state
#
#     elif wpt1[0] != wpt2[0]:
#         new_line0 = '{0} > DEFWPT {1},{2},{3}'.format(
#             time, wpt1, route_parameters[str( i )]['lat'], route_parameters[str( i )]['lon'] )
#         if state['action'] == 'cruise':
#             new_line1 = '{0} > ADDWPT {1} FLYOVER'.format( time, ac )
#             new_line2 = '{0} > ADDWPT {1} {2}, ,{3}'.format(
#                 time, ac, wpt1, str( route_parameters[str( i )]['speed'] * m_s2knot ) )
#             new_line3 = '{0} > {1} ATDIST {2} {3} SPD {4} {5}'.format(
#                 time, ac, wpt1, 0.015, ac, 5 )
#             new_line4 = '{0} > {1} AT {2} DO SPD {3} 0'.format(
#                 time, ac, wpt1, ac )
#             new_line5 = '{0} > {1} AT {2} DO ALT {3} {4}'.format(
#                 time, ac, wpt1, ac, route_parameters[str( i + 1 )]['alt'] * m2ft )
#             new_line6 = '{0} > {1} AT {2} DO VS {3} {4}'.format(
#                 time, ac, wpt1, ac, str( route_parameters[str( i )]['speed'] * m_s2ft_min ) )
#
#             scenario_file.write( new_line0 + '\n' + new_line1 + '\n' + new_line2 + '\n' +
#                                  new_line3 + '\n' + new_line4 + '\n' + new_line5 + '\n' +
#                                  new_line6 + '\n' )
#
#             state['ref_wpt'] = wpt1
#
#         elif state['action'] == 'climbing':
#             new_line0 = '{0} > {1} AT {2} DO {3} ATALT {4}, ALT {5} {6}'.format(
#                 time, ac, state['ref_wpt'], ac, route_parameters[str( i )]['alt'] * m2ft, ac,
#                 route_parameters[str( i + 1 )]['alt'] * m2ft )
#             new_line1 = '{0} > {1} AT {2} DO {3} ATALT {4}, VS {5} {6}'.format(
#                 time, ac, state['ref_wpt'], ac, route_parameters[str( i )]['alt'] * m2ft, ac,
#                 str( route_parameters[str( i )]['speed'] * m_s2ft_min ) )
#
#             scenario_file.write( new_line0 + '\n' + new_line1 + '\n' )
#
#         state['action'] = 'climbing'
#         state['heading'] = 0
#         return state

# DEPRECIATED
# def createInstruction( scenario_commands, route, i, ac, G, layers_dict, time, state ):
#     wpt1 = route[i]
#     wpt2 = route[i + 1]
#     m_s2knot = 1.944
#     m2ft = 3.281
#     m_s2ft_min = 197
#
#     if state['action'] is None:
#         new_line0 = '{0} > DEFWPT {1},{2},{3}'.format(
#             time, wpt1, G.nodes[wpt1]['y'], G.nodes[wpt1]['x'] )
# #         new_line1 = '{0} > DEFWPT {1},{2},{3}'.format(
# #             time, wpt2, G.nodes[wpt2]['y'], G.nodes[wpt2]['x'] )
#         if wpt1[0] == wpt2[0]:
#             hdg, increment = headingIncrement( 0, wpt1, wpt2, G )
#             new_line1 = '{0} > CRE {1} M600 {2} {3} {4} {5}'.format(
#                 time, ac, wpt1, hdg, layers_dict[wpt1[0]] * m2ft,
#                 str( G.edges[( wpt1, wpt2, 0 )]['speed'] * m_s2knot ) )
# #             new_line3 = '{0} > ADDWPT {1} {2}, , {3}'.format( time, ac, wpt2 )
# #             new_line4 = '{0} > SPD {1} {2}'.format(
# #                 time, ac, str( G.edges[( wpt1, wpt2, 0 )]['speed'] * m_s2knot ) )
#             state['action'] = 'cruise'
#             state['heading'] = hdg
#             aux = {}
#             aux['standard'] = new_line0 + '\n' + new_line1 + '\n'
#             aux['additional1'] = ''
#             aux['additional2'] = ''
#             scenario_commands[wpt1] = aux
#         else:
#             new_line1 = '{0} > CRE {1} M600 {2} 0 {3} 0'.format(
#                 time, ac, wpt1, layers_dict[wpt1[0]] * m2ft )
#             new_line2 = '{0} > ADDWPT {1} {2}, , {3}'.format(
#                 time, ac, wpt1, str( G.edges[( wpt1, wpt2, 0 )]['speed'] * m_s2knot ) )
#             new_line3 = '{0} > ALT {1} {2}'.format( time, ac, layers_dict[wpt2[0]] * m2ft )
#             new_line4 = '{0} > VS {1} {2}'.format(
#                 time, ac, str( G.edges[( wpt1, wpt2, 0 )]['speed'] * m_s2ft_min ) )
#             state['ref_wpt'] = wpt1
#             state['action'] = 'climbing'
#             state['heading'] = 0
#
#             aux = {}
#             aux['standard'] = new_line0 + '\n' + new_line1 + '\n' + new_line2 + '\n' + \
#                 new_line3 + '\n' + new_line4 + '\n'
#             aux['additional1'] = ''
#             aux['additional2'] = ''
#             scenario_commands[wpt1] = aux
#
#         return state, scenario_commands
#
#     if wpt1[0] == wpt2[0]:
#         new_heading, increment = headingIncrement( state['heading'], wpt1, wpt2, G )
#         new_line0 = '{0} > DEFWPT {1},{2},{3}'.format(
#             time, wpt1, G.nodes[wpt1]['y'], G.nodes[wpt1]['x'] )
#         if state['action'] == 'cruise':
#             turn_speed, turn_dist, turn_rad = turnDefinition( increment )
#             if turn_speed is None:
#                 new_line1 = '{0} > ADDWPT {1} FLYOVER'.format( time, ac )
#                 new_line2 = '{0} > ADDWPT {1} {2}, , {3}'.format(
#                     time, ac, wpt1, str( G.edges[( wpt1, wpt2, 0 )]['speed'] * m_s2knot ) )
#
#                 aux = {}
#                 aux['standard'] = new_line0 + '\n' + new_line1 + '\n' + new_line2 + '\n'
#                 aux['additional1'] = ''
#                 aux['additional2'] = ''
#                 scenario_commands[wpt1] = aux
#
#             else:
#                 new_line1 = '{0} > ADDWPT {1} FLYTURN'.format( time, ac )
#                 new_line2 = '{0} > ADDWPT {1} TURNSPD {2}'.format( time, ac, turn_speed )
#                 new_line3 = '{0} > ADDWPT {1} TURNRAD {2}'.format( time, ac, turn_rad )
#                 new_line4 = '{0} > ADDWPT {1} {2}, , {3}'.format(
#                     time, ac, wpt1, str( G.edges[( wpt1, wpt2, 0 )]['speed'] * m_s2knot ) )
#                 new_line5 = '{0} > {1} ATDIST {2} {3} SPD {4} {5}'.format(
#                     time, ac, wpt1, turn_dist, ac, turn_speed )
#                 new_line6 = '{0} > {1} AT {2} DO LNAV {3} ON'.format(
#                     time, ac, wpt1, ac )
#                 new_line7 = '{0} > {1} AT {2} DO VNAV {3} ON'.format(
#                     time, ac, wpt1, ac )
#                 aux = {}
#                 aux['standard'] = new_line0 + '\n' + new_line1 + '\n' + new_line2 + '\n' + \
#                     new_line3 + '\n' + new_line4 + '\n' + new_line5 + '\n'
#                 aux['additional1'] = ''
#                 aux['additional2'] = new_line6 + '\n' + new_line7 + '\n'
#                 scenario_commands[wpt1] = aux
#
#                 scenario_commands = turnDetection( route, i, turn_dist, G, scenario_commands )
#
#         elif state['action'] == 'climbing':
#             new_line1 = '{0} > {1} AT {2} DO {3} ATALT {4}, LNAV {5} ON'.format(
#                 time, ac, state['ref_wpt'], ac, layers_dict[wpt1[0]] * m2ft, ac )
#             new_line2 = '{0} > {1} AT {2} DO {3} ATALT {4}, VNAV {5} ON'.format(
#                 time, ac, state['ref_wpt'], ac, layers_dict[wpt1[0]] * m2ft, ac )
#             new_line3 = '{0} > {1} AT {2} DO {3} ATALT {4}, ADDWPT {5} {6}, {7}, {8}, {9}'.format(
#                 time, ac, state['ref_wpt'], ac, layers_dict[wpt1[0]] * m2ft, ac, wpt1, layers_dict[wpt1[0]] * m2ft,
#                 str( G.edges[( wpt1, wpt2, 0 )]['speed'] * m_s2knot ), state['ref_wpt'] )
#
#             aux = {}
#             aux['standard'] = new_line0 + '\n' + new_line1 + '\n' + new_line2 + '\n' + \
#                 new_line3 + '\n'
#             aux['additional'] = ''
#             scenario_commands[wpt1] = aux
#
#         state['action'] = 'cruise'
#         state['heading'] = new_heading
#         return state, scenario_commands
#
#     elif wpt1[0] != wpt2[0]:
#         new_line0 = '{0} > DEFWPT {1},{2},{3}'.format(
#             time, wpt1, G.nodes[wpt1]['y'], G.nodes[wpt1]['x'] )
#         if state['action'] == 'cruise':
#             new_line1 = '{0} > ADDWPT {1} FLYOVER'.format( time, ac )
#             new_line2 = '{0} > ADDWPT {1} {2}, ,{3}'.format(
#                 time, ac, wpt1, str( G.edges[( wpt1, wpt2, 0 )]['speed'] * m_s2knot ) )
#             new_line3 = '{0} > {1} ATDIST {2} {3} SPD {4} {5}'.format(
#                 time, ac, wpt1, 0.015, ac, 5 )
#             new_line4 = '{0} > {1} AT {2} DO SPD {3} 0'.format(
#                 time, ac, wpt1, ac )
#             new_line5 = '{0} > {1} AT {2} DO ALT {3} {4}'.format(
#                 time, ac, wpt1, ac, layers_dict[wpt2[0]] * m2ft )
#             new_line6 = '{0} > {1} AT {2} DO VS {3} {4}'.format(
#                 time, ac, wpt1, ac, str( G.edges[( wpt1, wpt2, 0 )]['speed'] * m_s2ft_min ) )
#
#             aux = {}
#             aux['standard'] = new_line0 + '\n' + new_line1 + '\n' + new_line2 + '\n' + \
#                 new_line3 + '\n' + new_line4 + '\n' + new_line5 + '\n' + new_line6 + '\n'
#             aux['additional'] = ''
#             scenario_commands[wpt1] = aux
#
#             state['ref_wpt'] = wpt1
#
#         elif state['action'] == 'climbing':
#             new_line0 = '{0} > {1} AT {2} DO {3} ATALT {4}, ALT {5} {6}'.format(
#                 time, ac, state['ref_wpt'], ac, layers_dict[wpt1[0]] * m2ft, ac, layers_dict[wpt2[0]] * m2ft )
#             new_line1 = '{0} > {1} AT {2} DO {3} ATALT {4}, VS {5} {6}'.format(
#                 time, ac, state['ref_wpt'], ac, layers_dict[wpt1[0]] * m2ft, ac,
#                 str( G.edges[( wpt1, wpt2, 0 )]['speed'] * m_s2ft_min ) )
#
#             aux = {}
#             aux['standard'] = new_line0 + '\n' + new_line1 + '\n'
#             aux['additional'] = ''
#             scenario_commands[wpt1] = aux
#
#         state['action'] = 'climbing'
#         state['heading'] = 0
#         return state, scenario_commands


def createFlightPlan( route, ac, departure_time, G, layers_dict, scenario_file ):
    """
    Create a flight plan for a drone. All the commands are written in a text file.

    Args:
            route (list): list of all waypoints of the route
            ac (dictionary): aircraft parameters {id, type, accel, v_max, vs_max}
            departure_time (string): string indicating the departure time
            G (graph)
            layers_dict (dictionary): dictionary with the information about layers and altitudes
            scenario_file (object): text file object where the commands are written

    """
    print( 'Creating flight plan of {0}...'.format( ac['id'] ) )
    state = {}
    route_parameters = routeParameters( G, route, ac )
    state['action'] = None
    for i in range( len( route ) - 1 ):
        state = createInstructionV3( 
            scenario_file, route_parameters, i, ac, G, layers_dict, departure_time, state )

    if state['action'] == 'cruise':
        m_s2knot = 1.944

        new_line0 = '{0} > DEFWPT {1},{2},{3}'.format( 
            departure_time, route[-1], G.nodes[route[-1]]['y'], G.nodes[route[-1]]['x'] )
        new_line1 = '{0} > ADDWPT {1} {2}, , {3}'.format( 
            departure_time, ac['id'], route[-1],
            str( route_parameters[str( len( route ) - 1 )]['speed'] * m_s2knot ) )
        new_line2 = '{0} > {1} ATDIST {2} 0.003 DEL {3}'.format( 
            departure_time, ac['id'], route[-1], ac['id'] )
        scenario_file.write( new_line0 + '\n' + new_line1 + '\n' + new_line2 + '\n' )
    elif state['action'] == 'climbing':
        m2ft = 3.281
        new_line0 = '{0} > {1} AT {2} DO {3} ATALT {4}, DEL {5}'.format( 
            departure_time, ac['id'], state['ref_wpt'], ac['id'], layers_dict[route[-1][0]] * m2ft, ac['id'] )
        scenario_file.write( new_line0 + '\n' )


def createDeliveryFlightPlan( route1, route2, ac, departure_time, G, layers_dict, scenario_file,
                              scenario_path, hovering_time=30 ):
    """
    Create a flight plan for a drone. All the commands are written in a text file.

    Args:
            route1 (list): list of all waypoints of the route to the delivery point
            route2 (list): list of all waypoints of the return route
            ac (dictionary): aircraft parameters {id, type, accel, v_max, vs_max}
            departure_time (string): string indicating the departure time
            G (graph)
            layers_dict (dictionary): dictionary with the information about layers and altitudes
            scenario_file (object): text file object where the commands are written
            scenario_path (string): string indicating the path of the delivery scenario
            hovering_time (integer): number of seconds the parcel takes to be delivered

    """
    print( 'Creating delivery flight plan of {0}...'.format( ac['id'] ) )
    return_path = scenario_path[:-4] + '_return.scn'
    return_path_rel = './' + '/'.join( scenario_path.split( '\\' )[4:] )
    m2ft = 3.281
    m_s2knot = 1.944
    m_s2ft_min = 197  # m/s to ft/min
    state = {}
    route_parameters1 = routeParameters( G, route1, ac )
    state['action'] = None
    for i in range( len( route1 ) - 1 ):
        state = createInstructionV3( 
            scenario_file, route_parameters1, i, ac, G, layers_dict, departure_time, state )

    if state['action'] == 'cruise':

        new_line0 = '{0} > DEFWPT {1},{2},{3}'.format( 
            departure_time, route1[-1], G.nodes[route1[-1]]['y'], G.nodes[route1[-1]]['x'] )
        new_line1 = '{0} > ADDWPT {1} {2}, , {3}'.format( 
            departure_time, ac['id'], route1[-1],
            str( route_parameters1[str( len( route1 ) - 2 )]['speed'] * m_s2knot ) )
        new_line2 = '{0} > {1} ATDIST {2} 0.03 SPD {3} 5'.format( 
            departure_time, ac['id'], route1[-1], ac['id'] )
        new_line3 = '{0} > {1} AT {2} DO {3} SPD 0'.format( 
            departure_time, ac['id'], route1[-1], ac['id'] )
        new_line4 = '{0} > {1} AT {2} DO {3} ATSPD 0, DELAY {4} DEL {5}'.format( 
            departure_time, ac['id'], route1[-1], ac['id'], str( hovering_time ), ac['id'] )
        new_line5 = '{0} > {1} AT {2} DO {3} ATSPD 0, DELAY {4} PCALL {5} REL '.format( 
            departure_time, ac['id'], route1[-1], ac['id'], str( hovering_time + 3 ), return_path_rel )

        scenario_file.write( new_line0 + '\n' + new_line1 + '\n' + new_line2 + '\n' + \
                             new_line3 + '\n' + new_line4 + '\n' + new_line5 + '\n' )
    elif state['action'] == 'climbing':

        new_line0 = '{0} > {1} AT {2} DO {3} ATALT {4}, VS {5} 0'.format( 
            departure_time, ac['id'], state['ref_wpt'], ac['id'], layers_dict[route1[-1][0]] * m2ft,
            ac['id'] )
        new_line1 = '{0} > {1} AT {2} DO {3} ATALT {6}, DELAY {4} DEL {5}'.format( 
            departure_time, ac['id'], route1[-1], ac['id'], str( hovering_time ), ac['id'],
            layers_dict[route1[-1][0]] * m2ft )
        new_line2 = '{0} > {1} AT {2} DO {3} ATALT {6}, DELAY {4} PCALL {5} REL'.format( 
            departure_time, ac['id'], route1[-1], ac['id'], str( hovering_time + 3 ), return_path_rel,
            layers_dict[route1[-1][0]] * m2ft )

        scenario_file.write( new_line0 + '\n' + new_line1 + '\n' + new_line2 + '\n' )

    scenario_file_return = open( return_path, 'w' )
    state2 = {}
    state2['action'] = None
    route_parameters2 = routeParameters( G, route2, ac )

    # Redefine departure time for the return trip - relative with respect to the initial departure
    departure_time = '00:00:00'

    for i in range( len( route2 ) - 1 ):
        state2 = createInstructionV3( 
            scenario_file_return, route_parameters2, i, ac, G, layers_dict, departure_time, state2 )

    if state2['action'] == 'cruise':

        new_line0 = '{0} > DEFWPT {1},{2},{3}'.format( 
            departure_time, route2[-1], G.nodes[route2[-1]]['y'], G.nodes[route2[-1]]['x'] )
        new_line1 = '{0} > ADDWPT {1} {2}, , {3}'.format( 
            departure_time, ac['id'], route2[-1],
            str( route_parameters2[str( len( route2 ) - 2 )]['speed'] * m_s2knot ) )
        new_line2 = '{0} > {1} ATDIST {2} 0.003 DEL {3}'.format( 
            departure_time, ac['id'], route2[-1], ac['id'] )
        scenario_file_return.write( new_line0 + '\n' + new_line1 + '\n' + new_line2 + '\n' )
    elif state2['action'] == 'climbing':
        new_line0 = '{0} > {1} AT {2} DO {3} ATALT {4}, DEL {5}'.format( 
            departure_time, ac['id'], state['ref_wpt'], ac['id'], layers_dict[route2[-1][0]] * m2ft, ac['id'] )
        scenario_file_return.write( new_line0 + '\n' )

    scenario_file_return.close()


def automaticFlightPlan( total_drones, base_name, G, layers_dict, scenario_general_path_base ):
    """
    Automatically create flight plans for a number of drones. It creates a scenario for each drone.
    In addition, it creates a general scenario that can be used to simulate all drone at the same
    time.

    Args:
            total_drones (integer): number of drones
            base_name (string): string indicating the base name of all drones (e.g. base_name = 'U',
                    drones are U1, U2,...
            G (graph)
            layers_dict (dictionary): dictionary with the information about layers and altitudes
            scenario_general_path_base (string): base path for the scenarios.
    """
    # General scenario that calls all drone scenarios
    scenario_general_path = scenario_general_path_base + '.scn'
    if not os.path.exists( os.path.dirname( scenario_general_path ) ):
        os.makedirs( os.path.dirname( scenario_general_path ) )
    scenario_general_file = open( scenario_general_path, 'w' )

    # Drone flight plan
    n = 1
    while n <= total_drones:
        orig_lat = random.uniform( 52.35, 52.4 )
        orig_lon = random.uniform( 9.72, 9.78 )
        dest_lat = random.uniform( 52.35, 52.4 )
        dest_lon = random.uniform( 9.72, 9.78 )

        if ox.distance.great_circle_vec( orig_lon, orig_lat, dest_lon, dest_lat ) < 2000:
            # We discard the trajectory if the origin and destination are too close
            continue

        orig = [orig_lon, orig_lat]
        dest = [dest_lon, dest_lat]

        name = base_name + str( n )  # drone name
        travel_time, route = trajectoryCalculation( G, orig, dest )

        print( 'The travel time of the route is {0}'.format( travel_time ) )
        print( 'The route is {0}'.format( route ) )

        # Path Planning
        ac = {'id': name, 'type': 'M600', 'accel': 3.5, 'v_max': 18, 'vs_max': 5 }
        departure_time = '00:00:00.00'
        scenario_path = scenario_general_path_base + '_test_' + str( n ) + '.scn'

        scenario_file = open( scenario_path, 'w' )
        createFlightPlan( route, ac, departure_time, G, layers_dict, scenario_file )
        scenario_file.close()

        scenario_general_file.write( '00:00:00.00 > PCALL ' + scenario_path + ' REL' + '\n' )

        n += 1

    scenario_general_file.close()


def addFlightData( orig_lat, orig_lon, orig_alt,
                  dest_lat, dest_lon, dest_alt,
                  departure_time_seconds,
                  drone_type,
                  purpose, data ):
    if departure_time_seconds < 36000:
            departure_time = '0{}'.format( str( datetime.timedelta( seconds=departure_time_seconds ) ) )
    else:
        departure_time = str( datetime.timedelta( seconds=departure_time_seconds ) )
    departure_time = departure_time[:11]

    data['origin_lat'].append( orig_lat )
    data['origin_lon'].append( orig_lon )
    data['origin_alt'].append( orig_alt )
    data['destination_lat'].append( dest_lat )
    data['destination_lon'].append( dest_lon )
    data['destination_alt'].append( dest_alt )
    data['departure'].append( departure_time )
    data['departure_s'].append( departure_time_seconds )
    data['drone'].append( drone_type )
    data['purpose'].append( purpose )


def createBackgroundTrafficCSV( density, avg_flight_duration, simulation_time, G, segments, config ):
    '''
    This function creates distributed origins and destinations for the background traffic in
    the city area defined in the configuration file

    Input:
        density - density desired
        avg_flight_duration - duration of the flights in average
        simulation time - duration of the simulation (seconds)
        G - city graph
        segments -
        config - configuration object
    Output:
        csv file
    '''
    # Data to be included in the CSV file
    data = { 'origin_lat': [], 'origin_lon': [], 'origin_alt': [],
             'destination_lat': [], 'destination_lon': [], 'destination_alt': [],
             'departure': [],
             'departure_s': [],
             'drone': [],
             'purpose': []}

    # Area of study
    mode = config['City'].get( 'mode' )

    if mode == 'rectangle':
        lat_min = config['City'].getfloat( 'hannover_lat_min' )
        lat_max = config['City'].getfloat( 'hannover_lat_max' )
        lon_min = config['City'].getfloat( 'hannover_lon_min' )
        lon_max = config['City'].getfloat( 'hannover_lon_max' )

        latitude_distance = ox.distance.great_circle_vec( lat_min, lon_min, lat_max, lon_min )
        longitude_distance = ox.distance.great_circle_vec( lat_min, lon_min, lat_min, lon_max )

        area = ( latitude_distance / 1000 ) * ( longitude_distance / 1000 )

    elif mode == 'square':
        raise ValueError( 'latitudes and longitudes min and max not implemented for the Square method!!' )
        zone_size = config['City'].getint( 'zone_size' )

        area = ( zone_size / 1000 ) ** 2

    flights_second = density * area / avg_flight_duration
    time_spacing = 1 / flights_second
    total_flights = flights_second * simulation_time

    # Drone flight plan
    min_flight_distance = 2500
    max_flight_distance = 20000
    # Consider that the background traffic includes different types of drones
    drone_type_distribution = { "M600": 0.5,
                                "Amzn": 0.3,
                                "W178": 0.2 }
    n = 1
    while n <= total_flights:

        orig_lat = random.uniform( lat_min, lat_max )
        orig_lon = random.uniform( lon_min, lon_max )
        dest_lat = random.uniform( lat_min, lat_max )
        dest_lon = random.uniform( lon_min, lon_max )

        if ox.distance.great_circle_vec( orig_lon, orig_lat, dest_lon, dest_lat ) < min_flight_distance:
            # We discard the trajectory if the origin and destination are too close
            continue
        elif ox.distance.great_circle_vec( orig_lon, orig_lat, dest_lon, dest_lat ) > max_flight_distance:
            # We discard the trajectory if the origin and destination are too far away
            continue

        # Check if the origin or destination is in a restricted area (spd = 0)
        if checkIfNoFlyZone( orig_lat, orig_lon, None, G, segments ):
            continue
        if checkIfNoFlyZone( dest_lat, dest_lon, None, G, segments ):
            continue

        print( 'Creating flight {0}...'.format( n ) )
        drone_type = random.choices( list( drone_type_distribution.keys() ),
                                     weights=tuple( drone_type_distribution.values() ) )[0]

        departure_time = '{}'.format( n * time_spacing )
        departure_time_seconds = n * time_spacing

        addFlightData( orig_lat, orig_lon, None,
                  dest_lat, dest_lon, None,
                  departure_time_seconds,
                  drone_type,
                  'background', data )

        n += 1
    # return data
    # CSV creation
    data_frame = pd.DataFrame( data )

    file_name = 'background_traffic_' + str( density ).replace( '.', '-' ) + '_' + str( int( avg_flight_duration ) ) + '_' + str( simulation_time ) + '.csv'

    path = sys.path[0] + '\\data\\' + file_name

    data_frame.to_csv( path )

    print( 'Background traffic stored in : {0}'.format( path ) )

def createDeliveryDrone( orig, dest, departure_time, frequency, uncertainty, distributed, simulation_time, data ):  # Add uncertainty / distribute
    ( orig_lat, orig_lon, orig_alt ) = orig
    ( dest_lat, dest_lon, dest_alt ) = dest

    if frequency != None:
        while departure_time < simulation_time:
            departure_time_aux = copy.deepcopy( departure_time )
            if uncertainty:
                departure_time += uncertainty
            if distributed:
                departure_time = random.randrange( departure_time, simulation_time )
            # Add one flight to data
            addFlightData( orig_lat, orig_lon, orig_alt,
                          dest_lat, dest_lon, dest_alt,
                          departure_time,
                          'M600',
                          'delivery', data )

            departure_time = departure_time_aux
            departure_time += frequency

    else:
        # Add one flight to data
        addFlightData( orig_lat, orig_lon, orig_alt,
                      dest_lat, dest_lon, dest_alt,
                      departure_time,
                      'M600',
                      'delivery', data )

def createDeliveryCSV( departure_times, frequencies, uncertainties, distributed, simulation_time ):
    '''
    This function creates the csv containing the data of the delivery drones

    Input:
        departure_times - list with 3 values []
                        - times must be integers (seconds)
                        - None for the drones that do not take part in the simulation
        frequencies - list with 3 values []
                    - None for the drones that only fly once
        uncertainties - list with 3 values []
                      - None for no uncertainty
        distributed - True to distribute the flights randomly in the range (departure, simulation_time)
                    - False
        simulation time - duration of the simulation (seconds)

    Output:
        csv
    '''
    # Data to be included in the CSV file
    data = { 'origin_lat': [], 'origin_lon': [], 'origin_alt': [],
             'destination_lat': [], 'destination_lon': [], 'destination_alt': [],
             'departure': [],
             'departure_s': [],
             'drone': [],
             'purpose': []}

    # Define the origin and destination points

    # (orig_1_lat, orig_1_lon, orig_1_alt) = (52.4216, 9.6704, 50)
    # (orig_2_lat, orig_2_lon, orig_2_alt) = (52.3145, 9.8171, 50)
    # (orig_3_lat, orig_3_lon, orig_3_alt) = (52.3701, 9.7422, 50)
    orig_1 = ( 52.4216, 9.6704, 50 )
    orig_2 = ( 52.3145, 9.8171, 50 )
    orig_3 = ( 52.3701, 9.7422, 50 )
    origins = [orig_1, orig_2, orig_3]

    # (dest_1_lat, dest_1_lon, dest_1_alt) = (52.3594, 9.7499, 25)
    dest_1 = ( 52.3594, 9.7499, 25 )

    # Checks the format
    idx = 0
    for time in departure_times:
        # if time is int:
        #    time = '0{}'.format( str( datetime.timedelta( seconds=time) ) )
        if time != None:
            print( 'Create flight plan orig {0}'.format( idx + 1 ) )
            createDeliveryDrone( origins[idx], dest_1, time, frequencies[idx], uncertainties[idx], distributed, simulation_time, data )
        idx += 1
    # CSV creation
    data_frame = pd.DataFrame( data )

    file_name = 'delivery_' + str( departure_times ).replace( '[', '' ).replace( ']', '' ).replace( ', ', '-' ) + '_' + \
        str( frequencies ).replace( '[', '' ).replace( ']', '' ).replace( ', ', '-' ) + '_' + \
        str( simulation_time ) + '.csv'

    path = sys.path[0] + '\\data\\' + file_name

    data_frame.to_csv( path )

    print( 'Delivery drones stored in : {0}'.format( path ) )

def createScenarioCSV( density, avg_flight_duration, departure_times, frequencies, simulation_time, G, segments, config ):
    '''
    This function creates distributed origins and destinations for the background traffic in
    the city area defined in the configuration file and for the delivery drones defined by the user

    Input:
        density - density desired
        avg_flight_duration - duration of the flights in average
        departure_times - list with 3 values []
                        - times must be integers (seconds)
                        - None for the drones that do not take part in the simulation
        frequencies - list with 3 values []
                    - None fro the drones that only fly once
        simulation time - duration of the simulation (seconds)
        config - configuration object

    Output:
        background csv file
        delivery csv file
    '''
    # Add background traffic
    print( 'Creating background traffic...' )
    createBackgroundTrafficCSV( density, avg_flight_duration, simulation_time, G, segments, config )

    # Add operation scenario (delivery)
    print( 'Creating delivery drones...' )
    createDeliveryCSV( departure_times, frequencies, simulation_time )



# def createAllDroneScenario( total_drones ):
#
#     departure_time = '00:00:00.00'
#     folder_path = r'C:\workspace3\bluesky\nommon\city_model\data\Drone_trajectory\scenario_1000_drones'
#     scenario_path = folder_path + '\scenario_base_' + str( total_drones ) + '.scn'
#     scenario_file = open( scenario_path, 'w' )
#     n = 1
#     while n <= total_drones:
#         new_line = '{0} > PCALL {1} REL'.format(
#             departure_time, folder_path + '\scenario_test_' + str( n ) + '.scn' )
#         scenario_file.write( new_line + '\n' )
#         n += 1
#     scenario_file.close()


def drawBuildings( config, scenario_path_base, time='00:00:00.00' ):
    """
    Creates the scenarios to represent the buildings in BlueSky. First, it loads the building data.
    Then creates several BlueSky scenarios. Each scenario prints the footprints of 10000 buildings.

    Args:
            config (configuration file): configuration file with all the relevant information
            scenario_path_base (string): base path for the scenarios.
            time (string): string indicating the time. Default value: '00:00:00.00'
    """
    directory = config['BuildingData']['directory_hannover']
    building_dict = readCity( directory )
    transformer = Transformer.from_crs( "EPSG:25832", "EPSG:4326", always_xy=True )
#     lon_min = config['BuildingData'].getfloat( 'lon_min' )
#     lon_max = config['BuildingData'].getfloat( 'lon_max' )
#     lat_min = config['BuildingData'].getfloat( 'lat_min' )
#     lat_max = config['BuildingData'].getfloat( 'lat_max' )
    name_base = 'poly'
    idx = 0
    scenario_idx = 1
    scenario_file = open( scenario_path_base + '_part' + str( scenario_idx ) + '.scn', 'w' )
    for building in building_dict:
        building_list = []
        centroid = building_dict[building]['centroid']
        centroid = transformer.transform( centroid[0], centroid[1] )
#         if ( centroid[0] > lon_min ) & ( centroid[0] < lon_max ) & ( centroid[1] > lat_min ) & ( centroid[1] < lat_max ) :
#             idx += 1
#         else:
#             continue
        idx += 1

        for point in building_dict[building]['footprint']:
            building_list += [transformer.transform( point[0], point[1] )]

        name = name_base + str( idx )
        new_line0 = '{0} > POLY {1}'.format( time, name )

        for point in building_list:
            new_line0 += ' '
            new_line0 += str( point[1] )
            new_line0 += ' '
            new_line0 += str( point[0] )

        scenario_file.write( new_line0 + '\n' )

        if idx % 10000 == 0:
            new_line1 = '{0} > ECHO {1}'.format( time, name )
            scenario_file.write( new_line1 + '\n' )
            scenario_file.close()
            scenario_idx += 1
            scenario_file = open( scenario_path_base + '_part' + str( scenario_idx ) + '.scn', 'w' )

    scenario_file.close()

def createFlightPlansFromCSV( default_path, path_csv, strategic_deconfliction, G, segments, config ):
    '''
    This function reads the data stored in a csv file and creates the flight plans accordingly
    '''

    # Open csv
    data_frame = pd.read_csv( path_csv )

    # Date
    date_string = datetime.datetime.now().strftime( "%Y%m%d_%H%M%S" )

    # Create the path where the scenarios are stored
    # scenario_general_path_base = path_csv[:-4]
    scenario_general_path_base = default_path + '\\scenario\\usepe\\exercise1\\' + \
        date_string + '_' + path_csv.split( '/' )[-1][:-4]

    # General scenario that calls all drone scenarios
    scenario_general_path = scenario_general_path_base + '\\' + 'scenario_master.scn'
    if not os.path.exists( os.path.dirname( scenario_general_path ) ):
        os.makedirs( os.path.dirname( scenario_general_path ) )
    scenario_general_file = open( scenario_general_path, 'w' )

    layers_dict = layersDict( config )

    for idx, row in data_frame.iterrows():

        # print(row)

        orig_lat = row['origin_lat']
        orig_lon = row['origin_lon']
        orig_alt = row['origin_alt']
        dest_lat = row['destination_lat']
        dest_lon = row['destination_lon']
        dest_alt = row['destination_alt']

        purpose = row['purpose']

        if purpose == 'background':
            orig = [orig_lon, orig_lat]
            dest = [dest_lon, dest_lat]
            # basename for the drones
            base_name = 'U'
            name = base_name + str( idx + 1 )  # drone name
            # Paths
            scenario_path = scenario_general_path_base + '\\' + 'scenario_background_traffic_drone' + str( idx + 1 ) + '.scn'
            relative_path = './usepe/exercise1/' + date_string + '_' + path_csv.split( '/' )[-1][:-4] + '/' + \
                'scenario_background_traffic_drone' + str( idx + 1 ) + '.scn'
        elif purpose == 'delivery':
            orig = [orig_lon, orig_lat, orig_alt]
            dest = [dest_lon, dest_lat, dest_alt]
            # basename for the drones
            base_name = 'D'
            name = base_name + str( idx + 1 )  # drone name
            # Paths
            scenario_path = scenario_general_path_base + '\\' + 'scenario_delivery_drone' + str( idx + 1 ) + '.scn'
            relative_path = './usepe/exercise1/' + date_string + '_' + path_csv.split( '/' )[-1][:-4] + '/' + \
                'scenario_delivery_drone' + str( idx + 1 ) + '.scn'

        # Get the drone features
        script_path = os.path.realpath( __file__ )
        drones_data_json_file = script_path[:-39] + '\\data\\performance\\OpenAP\\rotor\\aircraft.json'
        with open( drones_data_json_file ) as json_file:
            drone_features = json.load( json_file )

        ac = {'id': name,
              'type': row['drone'],
              'accel': 3.5,
              'v_max': drone_features[row['drone']]["envelop"]['v_max'],
              'vs_max': drone_features[row['drone']]["envelop"]['vs_max'],
              'safety_volume_size': 1}

        scenario_file = open( scenario_path, 'w' )
        if not strategic_deconfliction:
            if purpose == 'background':
                # Calculate optimum trajectory
                travel_time, route = trajectoryCalculation( G, orig, dest )
                # Path Planning
                departure_time = row['departure']
                createFlightPlan( route, ac, departure_time, G, layers_dict, scenario_file )
            elif purpose == 'delivery':
                # Calculate optimum trajectories (origin-delivery and delivery-origin)
                travel_time1, route1 = trajectoryCalculation( G, orig, dest )
                travel_time2, route2 = trajectoryCalculation( G, dest, orig )
                departure_time = row['departure']

                # Check scenario_file and scenario_path
                createDeliveryFlightPlan( route1, route2, ac, departure_time,
                                          G, layers_dict,
                                          scenario_file, scenario_path,
                                          hovering_time=30 )

        else:
            # TODO: finalise strategic deconfliction
            departure_time = row['departure_s']
            # initial_time = 0  # seconds
            # final_time = 2 * 3600  # seconds
            # users = initialPopulation( segments, initial_time, final_time )

            # users = deconflcitedScenario( orig, dest, ac, departure_time, G, users, initial_time,
            #                              final_time, segments, layers_dict, scenario_file, config )
        scenario_file.close()

        scenario_general_file.write( '00:00:00.00 > PCALL ' + relative_path + ' REL' + '\n' )
    scenario_general_file.close()
    print( 'Output scenario stored in the following directory {0}'.format( scenario_general_path_base ) )




if __name__ == '__main__':
    pass
    # config = configparser.ConfigParser()
    # config_path = "C:/workspace3/bluesky/nommon/city_model/settings.cfg"
    # config.read( config_path )
    # """
    # # Drawing buildings
    # time = '00:00:00.00'
    # scenario_path = r'C:\workspace3\bluesky\nommon\city_model\scenario_buildings.scn'
    # scenario_file = open( scenario_path, 'w' )
    # drawBuildings( config, scenario_file, time )
    # scenario_file.close()
    # """
    # createAllDroneScenario( 100 )
    #
    # print( 'Finish' )
