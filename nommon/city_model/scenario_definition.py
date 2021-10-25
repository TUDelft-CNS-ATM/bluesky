#!/usr/bin/python

"""

"""
import configparser
import math
import os
import random
import string

from pyproj import Transformer

from nommon.city_model.building_height import readCity
from nommon.city_model.multi_di_graph_3D import MultiDiGrpah3D
from nommon.city_model.path_planning import trajectoryCalculation
from nommon.city_model.utils import read_my_graphml
import osmnx as ox


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
    if new_heading == 360:
        new_heading = new_heading - 360
    increment = abs( new_heading - actual_heading )
    return new_heading, increment


def turnDefinition( increment ):
    """
    Compute some parameters needed to control the change of direction of the drone. It returns the
    turn speed, the turn distance and turn radius.

    Args:
            increment (float): float indicating the change of heading

    Returns:
            turn_speed (float or None): indicating the velocity when performing the turn
            turn_dist (float or None): indicating the distance at which the drone has to start to
            decelerate
            turn_rad (float or None): indicating the turn radius
    """
    if increment < 20:
        turn_speed = None
        turn_dist = None
        turn_rad = None
    elif increment < 45:
        turn_speed = 20
        turn_dist = 0.01
        turn_rad = 0.0003
    elif increment < 60:
        turn_speed = 17
        turn_dist = 0.011
        turn_rad = 0.0003
    elif increment < 70:
        turn_speed = 15
        turn_dist = 0.013
        turn_rad = 0.0003
    elif increment < 80:
        turn_speed = 10
        turn_dist = 0.015
        turn_rad = 0.0003
    elif increment < 90:
        turn_speed = 8
        turn_dist = 0.017
        turn_rad = 0.0002
    else:
        turn_speed = 5
        turn_dist = 0.021
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


def routeParameters( G, route ):
    """
    Compute all the information about the route (e.g. turn distance, turn speed, altitude, etc.). It
    is stored as a dictionary.

    Args:
            G (grah)
            route (list): list of waypoints

    Returns:
            route_parameters (dictionary): dictionary with all the information about the route
    """
    route_parameters = {}
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
            node['speed'] = G.edges[( name, route[i + 1], 0 )]['speed']
            node['dist'] = G.edges[( name, route[i + 1], 0 )]['length']
        else:
            new_heading, increment = headingIncrement( route_parameters[str( i - 1 )]['hdg'], name,
                                                       route[i + 1], G )
            turn_speed, turn_dist, turn_rad = turnDefinition( increment )
            node['name'] = name
            node['lat'] = G.nodes[name]['y']
            node['lon'] = G.nodes[name]['x']
            node['alt'] = G.nodes[name]['z']
            node['turn speed'] = turn_speed
            node['turn rad'] = turn_rad
            node['turn dist'] = turn_dist
            node['hdg'] = new_heading
            node['speed'] = G.edges[( name, route[i + 1], 0 )]['speed']
            node['dist'] = G.edges[( name, route[i + 1], 0 )]['length']
        route_parameters[str( i )] = node

        final_node = {}
        final_node['name'] = route[-1]
        final_node['lat'] = G.nodes[route[-1]]['y']
        final_node['lon'] = G.nodes[route[-1]]['x']
        final_node['alt'] = G.nodes[route[-1]]['z']
        final_node['turn speed'] = None
        final_node['turn rad'] = None
        final_node['turn dist'] = None
        final_node['hdg'] = None
        final_node['speed'] = None
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
            ac (string): aircraft name
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
            new_line1 = '{0} > CRE {1} M600 {2} {3} {4} {5} {6}'.format( 
                time, ac, route_parameters[str( i )]['lat'], route_parameters[str( i )]['lon'],
                hdg, route_parameters[str( i )]['alt'] * m2ft,
                str( G.edges[( wpt1, wpt2, 0 )]['speed'] * m_s2knot ) )

            state['action'] = 'cruise'
            state['heading'] = hdg

            scenario_file.write( new_line1 + '\n' )
        else:
            new_line0 = '{0} > DEFWPT {1},{2},{3}'.format( 
                time, wpt1, route_parameters[str( i )]['lat'], route_parameters[str( i )]['lon'] )
            new_line1 = '{0} > CRE {1} M600 {2} 0 {3} 0'.format( 
                time, ac, wpt1, route_parameters[str( i )]['alt'] * m2ft )
            new_line2 = '{0} > ADDWPT {1} {2}, , {3}'.format( 
                time, ac, wpt1, str( route_parameters[str( i )]['speed'] * m_s2knot ) )
            new_line3 = '{0} > ALT {1} {2}'.format( time, ac,
                                                    route_parameters[str( i + 1 )]['alt'] * m2ft )
            new_line4 = '{0} > VS {1} {2}'.format( 
                time, ac, str( route_parameters[str( i )]['speed'] * m_s2ft_min ) )
            state['ref_wpt'] = wpt1
            state['action'] = 'climbing'
            state['heading'] = 0

            scenario_file.write( new_line0 + '\n' + new_line1 + '\n' + new_line2 + '\n' +
                                 new_line3 + '\n' + new_line4 + '\n' )

        return state

    if wpt1[0] == wpt2[0]:  # wpt1 and wpt2 in the same layer
        if state['action'] == 'cruise':  # drone was flying horizontally
            turn_speed = route_parameters[str( i )]['turn speed']
            turn_dist = route_parameters[str( i )]['turn dist']
            turn_rad = route_parameters[str( i )]['turn rad']
            if turn_speed is None:
                new_line1 = '{0} > ADDWPT {1} FLYOVER'.format( time, ac )
                new_line2 = '{0} > ADDWPT {1} {2} {3}, , {4}'.format( 
                    time, ac, route_parameters[str( i )]['lat'], route_parameters[str( i )]['lon'],
                    str( route_parameters[str( i )]['speed'] * m_s2knot ) )

                scenario_file.write( new_line1 + '\n' + new_line2 + '\n' )

            else:
                option = turnDetectionV2( route_parameters, i )
                if option == 1:
                    new_line0 = '{0} > DEFWPT {1},{2},{3}'.format( 
                        time, wpt1, route_parameters[str( i )]['lat'], route_parameters[str( i )]['lon'] )
                    new_line1 = '{0} > ADDWPT {1} FLYTURN'.format( time, ac )
                    new_line2 = '{0} > ADDWPT {1} TURNSPD {2}'.format( time, ac, turn_speed )
                    new_line3 = '{0} > ADDWPT {1} TURNRAD {2}'.format( time, ac, turn_rad )
                    new_line4 = '{0} > ADDWPT {1} {2}, , {3}'.format( 
                        time, ac, wpt1, str( route_parameters[str( i )]['speed'] * m_s2knot ) )
                    new_line5 = '{0} > {1} ATDIST {2} {3} SPD {4} {5}'.format( 
                        time, ac, wpt1, turn_dist, ac, turn_speed )
                    new_line6 = '{0} > {1} AT {2} DO LNAV {3} ON'.format( 
                        time, ac, wpt1, ac )
                    new_line7 = '{0} > {1} AT {2} DO VNAV {3} ON'.format( 
                        time, ac, wpt1, ac )

                    scenario_file.write( new_line0 + '\n' + new_line1 + '\n' + new_line2 + '\n' +
                                         new_line3 + '\n' + new_line4 + '\n' + new_line5 + '\n' +
                                         new_line6 + '\n' + new_line7 + '\n' )
                elif option == 2:
                    new_line0 = '{0} > DEFWPT {1},{2},{3}'.format( 
                        time, wpt1, route_parameters[str( i )]['lat'], route_parameters[str( i )]['lon'] )
                    new_line1 = '{0} > ADDWPT {1} FLYTURN'.format( time, ac )
                    new_line2 = '{0} > ADDWPT {1} TURNSPD {2}'.format( time, ac, turn_speed )
                    new_line3 = '{0} > ADDWPT {1} TURNRAD {2}'.format( time, ac, turn_rad )
                    new_line4 = '{0} > ADDWPT {1} {2}, , {3}'.format( 
                        time, ac, wpt1, str( route_parameters[str( i )]['speed'] * m_s2knot ) )
                    new_line5 = '{0} > {1} ATDIST {2} {3} SPD {4} {5}'.format( 
                        time, ac, wpt1, turn_dist, ac, turn_speed )

                    scenario_file.write( new_line0 + '\n' + new_line1 + '\n' + new_line2 + '\n' +
                                         new_line3 + '\n' + new_line4 + '\n' + new_line5 + '\n' )
                elif option == 3:
                    new_line1 = '{0} > ADDWPT {1} FLYTURN'.format( time, ac )
                    new_line2 = '{0} > ADDWPT {1} TURNSPD {2}'.format( time, ac, turn_speed )
                    new_line3 = '{0} > ADDWPT {1} TURNRAD {2}'.format( time, ac, turn_rad )
                    new_line4 = '{0} > ADDWPT {1} {2} {3}, , {4}'.format( 
                        time, ac, route_parameters[str( i )]['lat'],
                        route_parameters[str( i )]['lon'],
                        str( route_parameters[str( i )]['speed'] * m_s2knot ) )

                    scenario_file.write( new_line1 + '\n' + new_line2 + '\n' +
                                         new_line3 + '\n' + new_line4 + '\n' )

        elif state['action'] == 'climbing':  # drone was climbing
            new_line1 = '{0} > {1} AT {2} DO {3} ATALT {4}, LNAV {5} ON'.format( 
                time, ac, state['ref_wpt'], ac, route_parameters[str( i )]['alt'] * m2ft, ac )
            new_line2 = '{0} > {1} AT {2} DO {3} ATALT {4}, VNAV {5} ON'.format( 
                time, ac, state['ref_wpt'], ac, route_parameters[str( i )]['alt'] * m2ft, ac )
            new_line3 = '{0} > {1} AT {2} DO {3} ATALT {4}, ADDWPT {5} {6} {7}, {8}, {9}, {10}'.format( 
                time, ac, state['ref_wpt'], ac, route_parameters[str( i )]['alt'] * m2ft, ac,
                route_parameters[str( i )]['lat'], route_parameters[str( i )]['lon'],
                route_parameters[str( i )]['alt'] * m2ft,
                str( route_parameters[str( i )]['speed'] * m_s2knot ), state['ref_wpt'] )

            scenario_file.write( new_line1 + '\n' + new_line2 + '\n' +
                                 new_line3 + '\n' )

        state['action'] = 'cruise'
        state['heading'] = route_parameters[str( i )]['hdg']
        return state

    elif wpt1[0] != wpt2[0]:  # wpt1 and wpt2 at different altitude
        new_line0 = '{0} > DEFWPT {1},{2},{3}'.format( 
            time, wpt1, route_parameters[str( i )]['lat'], route_parameters[str( i )]['lon'] )
        if state['action'] == 'cruise':  # drone was flying horizontally
            new_line1 = '{0} > ADDWPT {1} FLYOVER'.format( time, ac )
            new_line2 = '{0} > ADDWPT {1} {2}, ,{3}'.format( 
                time, ac, wpt1, str( route_parameters[str( i )]['speed'] * m_s2knot ) )
            new_line3 = '{0} > {1} ATDIST {2} {3} SPD {4} {5}'.format( 
                time, ac, wpt1, 0.015, ac, 5 )
            new_line4 = '{0} > {1} AT {2} DO SPD {3} 0'.format( 
                time, ac, wpt1, ac )
            new_line5 = '{0} > {1} AT {2} DO ALT {3} {4}'.format( 
                time, ac, wpt1, ac, route_parameters[str( i + 1 )]['alt'] * m2ft )
            new_line6 = '{0} > {1} AT {2} DO VS {3} {4}'.format( 
                time, ac, wpt1, ac, str( route_parameters[str( i )]['speed'] * m_s2ft_min ) )

            scenario_file.write( new_line0 + '\n' + new_line1 + '\n' + new_line2 + '\n' +
                                 new_line3 + '\n' + new_line4 + '\n' + new_line5 + '\n' +
                                 new_line6 + '\n' )

            state['ref_wpt'] = wpt1

        elif state['action'] == 'climbing':  # drone was climbing
            new_line0 = '{0} > {1} AT {2} DO {3} ATALT {4}, ALT {5} {6}'.format( 
                time, ac, state['ref_wpt'], ac, route_parameters[str( i )]['alt'] * m2ft, ac,
                route_parameters[str( i + 1 )]['alt'] * m2ft )
            new_line1 = '{0} > {1} AT {2} DO {3} ATALT {4}, VS {5} {6}'.format( 
                time, ac, state['ref_wpt'], ac, route_parameters[str( i )]['alt'] * m2ft, ac,
                str( route_parameters[str( i )]['speed'] * m_s2ft_min ) )

            scenario_file.write( new_line0 + '\n' + new_line1 + '\n' )

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
            ac (string): aircraft name
            departure_time (string): string indicating the departure time
            G (graph)
            layers_dict (dictionary): dictionary with the information about layers and altitudes
            scenario_file (object): text file object where the commands are written

    """
    print( 'Creating flight plan of {0}...'.format( ac ) )
    state = {}
    route_parameters = routeParameters( G, route )
    state['action'] = None
    for i in range( len( route ) - 1 ):
        state = createInstructionV3( 
            scenario_file, route_parameters, i, ac, G, layers_dict, departure_time, state )

    if state['action'] == 'cruise':
        m_s2knot = 1.944
        new_line0 = '{0} > DEFWPT {1},{2},{3}'.format( 
            departure_time, route[-1], G.nodes[route[-1]]['y'], G.nodes[route[-1]]['x'] )
        new_line1 = '{0} > ADDWPT {1} {2}, , {3}'.format( 
            departure_time, ac, route[-1],
            str( G.edges[( route[-2], route[-1], 0 )]['speed'] * m_s2knot ) )
        new_line2 = '{0} > {1} ATDIST {2} 0.001 DEL {3}'.format( 
            departure_time, ac, route[-1], ac )
        scenario_file.write( new_line0 + '\n' + new_line1 + '\n' + new_line2 + '\n' )
    elif state['action'] == 'climbing':
        m2ft = 3.281
        new_line0 = '{0} > {1} AT {2} DO {3} ATALT {4}, DEL {5}'.format( 
            departure_time, ac, state['ref_wpt'], ac, layers_dict[route[-1][0]] * m2ft, ac )
        scenario_file.write( new_line0 + '\n' )


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
        ac = name
        departure_time = '00:00:00.00'
        scenario_path = scenario_general_path_base + '_test_' + str( n ) + '.scn'

        scenario_file = open( scenario_path, 'w' )
        createFlightPlan( route, ac, departure_time, G, layers_dict, scenario_file )
        scenario_file.close()

        scenario_general_file.write( '00:00:00.00 > PCALL ' + scenario_path + ' REL' + '\n' )

        n += 1

    scenario_general_file.close()


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
