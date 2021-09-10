#!/usr/bin/python

"""

"""
import configparser
import math
import string
from pyproj import Transformer

from nommon.city_model.auxiliar import read_my_graphml
from nommon.city_model.building_height import readCity
from nommon.city_model.multi_di_graph_3D import MultiDiGrpah3D


__author__ = 'jbueno'
__copyright__ = '(c) Nommon 2021'


def headingIncrement( actual_heading, wpt1, wpt2, G ):
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


def createWaypoint( scenario_path, G ):
    print( 'Creating waypoints...' )
    scenario_file = open( scenario_path, 'a' )
    for n in G:
        new_line = '00:00:00.00 > DEFWPT {0},{1},{2}'.format( n, G.nodes[n]['y'], G.nodes[n]['x'] )
        scenario_file.write( new_line + '\n' )

    scenario_file.close()


def turnDefinition( increment ):
    if increment < 20:
        turn_speed = None
        turn_dist = None
    elif increment < 60:
        turn_speed = 20
        turn_dist = 0.01
    elif increment < 90:
        turn_speed = 10
        turn_dist = 0.015
    else:
        turn_speed = 5
        turn_dist = 0.02

    return turn_speed, turn_dist


def createInstruction( scenario_file, wpt1, wpt2, ac, G, layers_dict, time, state ):
    m_s2knot = 1.944
    m2ft = 3.281
    m_s2ft_min = 197

    if state['action'] is None:
        new_line0 = '{0} > DEFWPT {1},{2},{3}'.format( 
            time, wpt1, G.nodes[wpt1]['y'], G.nodes[wpt1]['x'] )
#         new_line1 = '{0} > DEFWPT {1},{2},{3}'.format(
#             time, wpt2, G.nodes[wpt2]['y'], G.nodes[wpt2]['x'] )
        if wpt1[0] == wpt2[0]:
            hdg, increment = headingIncrement( 0, wpt1, wpt2, G )
            new_line1 = '{0} > CRE {1} M600 {2} {3} {4} {5}'.format( 
                time, ac, wpt1, hdg, layers_dict[wpt1[0]] * m2ft,
                str( G.edges[( wpt1, wpt2, 0 )]['speed'] * m_s2knot ) )
#             new_line3 = '{0} > ADDWPT {1} {2}, , {3}'.format( time, ac, wpt2 )
#             new_line4 = '{0} > SPD {1} {2}'.format(
#                 time, ac, str( G.edges[( wpt1, wpt2, 0 )]['speed'] * m_s2knot ) )
            state['action'] = 'cruise'
            state['heading'] = hdg
            scenario_file.write( new_line0 + '\n' + new_line1 + '\n' )
        else:
            new_line1 = '{0} > CRE {1} M600 {2} 0 {3} 0'.format( 
                time, ac, wpt1, layers_dict[wpt1[0]] * m2ft )
            new_line2 = '{0} > ADDWPT {1} {2}, , {3}'.format( 
                time, ac, wpt1, str( G.edges[( wpt1, wpt2, 0 )]['speed'] * m_s2knot ) )
            new_line3 = '{0} > ALT {1} {2}'.format( time, ac, layers_dict[wpt2[0]] * m2ft )
            new_line4 = '{0} > VS {1} {2}'.format( 
                time, ac, str( G.edges[( wpt1, wpt2, 0 )]['speed'] * m_s2ft_min ) )
            state['ref_wpt'] = wpt1
            state['action'] = 'climbing'
            state['heading'] = 0
            scenario_file.write( new_line0 + '\n' + new_line1 + '\n' + new_line2 + '\n' +
                new_line3 + '\n' + new_line4 + '\n' )

        return state

    if wpt1[0] == wpt2[0]:
        new_heading, increment = headingIncrement( state['heading'], wpt1, wpt2, G )
        new_line0 = '{0} > DEFWPT {1},{2},{3}'.format( 
            time, wpt1, G.nodes[wpt1]['y'], G.nodes[wpt1]['x'] )
        if state['action'] == 'cruise':
            turn_speed, turn_dist = turnDefinition( increment )
            if turn_speed is None:
                new_line1 = '{0} > ADDWPT {1} FLYOVER'.format( time, ac )
                new_line2 = '{0} > ADDWPT {1} {2}, , {3}'.format( 
                    time, ac, wpt1, str( G.edges[( wpt1, wpt2, 0 )]['speed'] * m_s2knot ) )
                scenario_file.write( new_line0 + '\n' + new_line1 + '\n' + new_line2 + '\n' )
            else:
                new_line1 = '{0} > ADDWPT {1} FLYTURN'.format( time, ac )
                new_line2 = '{0} > ADDWPT {1} TURNSPD {2}'.format( time, ac, turn_speed )
                new_line3 = '{0} > ADDWPT {1} {2}, , {3}'.format( 
                    time, ac, wpt1, str( G.edges[( wpt1, wpt2, 0 )]['speed'] * m_s2knot ) )
                new_line4 = '{0} > {1} ATDIST {2} {3} SPD {4} {5}'.format( 
                    time, ac, wpt1, turn_dist, ac, turn_speed )
                new_line5 = '{0} > {1} AT {2} DO LNAV {3} ON'.format( 
                    time, ac, wpt1, ac )
                new_line6 = '{0} > {1} AT {2} DO VNAV {3} ON'.format( 
                    time, ac, wpt1, ac )
#                 scenario_file.write( new_line0 + '\n' + new_line1 + '\n' + new_line2 + '\n' +
#                     new_line3 + '\n' )
                scenario_file.write( new_line0 + '\n' + new_line1 + '\n' + new_line2 + '\n' +
                                     new_line3 + '\n' + new_line4 + '\n' + new_line5 + '\n' +
                                     new_line6 + '\n' )
#             new_line1 = '{0} > {1} AT {2} DO SPD {3} {4}'.format(
#                 time, ac, wpt1, ac, str( G.edges[( wpt1, wpt2, 0 )]['speed'] * m_s2knot ) )
        elif state['action'] == 'climbing':
            new_line1 = '{0} > {1} AT {2} DO {3} ATALT {4}, LNAV {5} ON'.format( 
                time, ac, state['ref_wpt'], ac, layers_dict[wpt1[0]] * m2ft, ac )
            new_line2 = '{0} > {1} AT {2} DO {3} ATALT {4}, VNAV {5} ON'.format( 
                time, ac, state['ref_wpt'], ac, layers_dict[wpt1[0]] * m2ft, ac )
            new_line3 = '{0} > {1} AT {2} DO {3} ATALT {4}, ADDWPT {5} {6}, {7}, {8}, {9}'.format( 
                time, ac, state['ref_wpt'], ac, layers_dict[wpt1[0]] * m2ft, ac, wpt1, layers_dict[wpt1[0]] * m2ft,
                str( G.edges[( wpt1, wpt2, 0 )]['speed'] * m_s2knot ), state['ref_wpt'] )
            scenario_file.write( new_line0 + '\n' + new_line1 + '\n' + new_line2 + '\n' +
                                 new_line3 + '\n' )

        state['action'] = 'cruise'
        state['heading'] = new_heading
        return state

    elif wpt1[0] != wpt2[0]:
        new_line0 = '{0} > DEFWPT {1},{2},{3}'.format( 
            time, wpt1, G.nodes[wpt1]['y'], G.nodes[wpt1]['x'] )
        if state['action'] == 'cruise':
            new_line1 = '{0} > ADDWPT {1} FLYOVER'.format( time, ac )
            new_line2 = '{0} > ADDWPT {1} {2}, ,{3}'.format( 
                time, ac, wpt1, str( G.edges[( wpt1, wpt2, 0 )]['speed'] * m_s2knot ) )
            new_line3 = '{0} > {1} ATDIST {2} {3} SPD {4} {5}'.format( 
                time, ac, wpt1, 0.015, ac, 5 )
            new_line4 = '{0} > {1} AT {2} DO SPD {3} 0'.format( 
                time, ac, wpt1, ac )
            new_line5 = '{0} > {1} AT {2} DO ALT {3} {4}'.format( 
                time, ac, wpt1, ac, layers_dict[wpt2[0]] * m2ft )
            new_line6 = '{0} > {1} AT {2} DO VS {3} {4}'.format( 
                time, ac, wpt1, ac, str( G.edges[( wpt1, wpt2, 0 )]['speed'] * m_s2ft_min ) )

            scenario_file.write( new_line0 + '\n' + new_line1 + '\n' + new_line2 + '\n' +
                                 new_line3 + '\n' + new_line4 + '\n' + new_line5 + '\n' +
                                 new_line6 + '\n' )

            state['ref_wpt'] = wpt1

        elif state['action'] == 'climbing':
            new_line0 = '{0} > {1} AT {2} DO {3} ATALT {4}, ALT {5} {6}'.format( 
                time, ac, state['ref_wpt'], ac, layers_dict[wpt1[0]] * m2ft, ac, layers_dict[wpt2[0]] * m2ft )
            new_line1 = '{0} > {1} AT {2} DO {3} ATALT {4}, VS {5} {6}'.format( 
                time, ac, state['ref_wpt'], ac, layers_dict[wpt1[0]] * m2ft, ac,
                str( G.edges[( wpt1, wpt2, 0 )]['speed'] * m_s2ft_min ) )

            scenario_file.write( new_line0 + '\n' + new_line1 + '\n' )

        state['action'] = 'climbing'
        state['heading'] = 0
        return state


def createFlightPlan( route, ac, departure_time, G, layers_dict, scenario_file ):
    print( 'Creating flight plan of {0}...'.format( ac ) )
    state = {}
    state['action'] = None
    for i in range( len( route ) - 1 ):
        state = createInstruction( scenario_file, route[i], route[i + 1], ac, G, layers_dict,
                                   departure_time, state )

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


def drawBuildings( config, scenario_path_base, time='00:00:00.00' ):
    directory = config['BuildingData']['directory_hannover']
#     directory = r"C:\Users\jbueno\Desktop\Stadtmodell_Hannover_CityGML_LoD1\Tests"
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
#             lat0 = str( building_list[i][1] )
#             lon0 = str( building_list[i][0] )
#
#             if i == len( building_list ) - 1:
#                 lat1 = str( building_list[0][1] )
#                 lon1 = str( building_list[0][0] )
#             else:
#                 lat1 = str( building_list[i + 1][1] )
#                 lon1 = str( building_list[i + 1][0] )
#             new_line0 = '{0} > Line, {1},{2},{3},{4},{5}'.format(
#                 time, name, lat0, lon0, lat1, lon1 )

        scenario_file.write( new_line0 + '\n' )

        if idx % 10000 == 0:
            new_line1 = '{0} > ECHO {1}'.format( time, name )
            scenario_file.write( new_line1 + '\n' )
            scenario_file.close()
            scenario_idx += 1
            scenario_file = open( scenario_path_base + '_part' + str( scenario_idx ) + '.scn', 'w' )

    scenario_file.close()


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config_path = "C:/workspace3/bluesky/nommon/city_model/settings.cfg"
    config.read( config_path )

    # Drawing buildings
    time = '00:00:00.00'
    scenario_path = r'C:\workspace3\bluesky\nommon\city_model\scenario_buildings.scn'
    scenario_file = open( scenario_path, 'w' )
    drawBuildings( config, scenario_file, time )
    scenario_file.close()
#

    print( 'Finish' )
