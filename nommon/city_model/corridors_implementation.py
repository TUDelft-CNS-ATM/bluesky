#!/usr/bin/python

"""

"""
from builtins import int
from distutils.command.config import config
import configparser
import csv
import math
import os

from dynamic_segments import defineSegment
import osmnx as ox


__author__ = 'jbueno'
__copyright__ = '(c) Nommon 2021'


def insertionNode( G, lon, lat, altitude ):
    # nearest_node = ox.distance.nearest_nodes( G, X=lon, Y=lat )

    nodes = list( G.nodes )
    # nearest_latlon = list( filter( lambda node: str( node )[1:] == nearest_node[1:]), nodes )
    nearest_latlon = list( filter( lambda node: str( node )[:3] != 'COR' , nodes ) )

    nearest_node = nearest_latlon[0]
    delta_z = abs( ( G.nodes[nearest_node]['z'] - altitude ) ** 2 +
                  ( G.nodes[nearest_node]['y'] - lat ) ** 2 +
                  ( G.nodes[nearest_node]['x'] - lon ) ** 2 )

    for node in nearest_latlon[1:]:
        delta_z_aux = abs( ( G.nodes[node]['z'] - altitude ) ** 2 +
                  ( G.nodes[node]['y'] - lat ) ** 2 +
                  ( G.nodes[node]['x'] - lon ) ** 2 )
        if delta_z_aux < delta_z:
            delta_z = delta_z_aux
            nearest_node = node
    return nearest_node

def latLon2XY( lat, lon, lat_0 ):
    '''

    '''

def entryNodes( G, segments, node, name, speed, next_node, config ):
    n_layers = config['Layers'].getint( 'number_of_layers' )
    layer_width = config['Layers'].getint( 'layer_width' )
    entry_low_height = n_layers * layer_width
    # Get longitude and latitude for the acceleration entry point
    lon = G.nodes[node]['x']
    lat = G.nodes[node]['y']
    lon_next = G.nodes[next_node]['x']
    lat_next = G.nodes[next_node]['y']

    # transform latitude, longitude to x, y
    r = 6378137
    x = r * math.radians( lon ) * math.cos( math.radians( lat ) )
    y = r * math.radians( lat )
    x_next = r * math.radians( lon_next ) * math.cos( math.radians( lat ) )
    y_next = r * math.radians( lat_next )

    # Unit vector opposite to the next point
    mod = math.sqrt( ( y_next - y ) ** 2 + ( x_next - x ) ** 2 )
    unit_vector_x = ( x - x_next ) / mod
    unit_vector_y = ( y - y_next ) / mod
    entry_angle = math.atan2( unit_vector_y, unit_vector_x ) + math.pi / 6

    # Entry point coordinates
    acceleration_lenght = config['Corridors'].getint( 'acceleration_length' )
    entry_x = x + math.cos( entry_angle ) * acceleration_lenght
    entry_y = y + math.sin( entry_angle ) * acceleration_lenght
    entry_lon = math.degrees( entry_x / ( r * math.cos( math.radians( lat ) ) ) )
    entry_lat = math.degrees( entry_y / r )
    # Adding nodes
    node_low = node + '_entry_1'
    node_high = node + '_entry_2'
    G.addNodeAltitude( node_low, entry_lat, entry_lon, entry_low_height )
    G.addNodeAltitude( node_high, entry_lat, entry_lon, G.nodes[node]['z'] )
    # Adding edges for those nodes
    G.add_edge( node_low, node_high, 0, oneway=False, segment='new',
                    speed=speed, length=G.nodes[node]['z'] - entry_low_height )
    G.add_edge( node_high, node_low, 0, oneway=False, segment=name,
                    speed=speed, length=G.nodes[node]['z'] - entry_low_height )

    G.add_edge( node_high, node, 0, oneway=False, segment='new', speed=speed,
               length=ox.distance.great_circle_vec( entry_lat, entry_lon,
                                                         G.nodes[node]['y'],
                                                         G.nodes[node]['x'] ) )
    G.add_edge( node, node_high, 0, oneway=False, segment=name, speed=speed,
               length=ox.distance.great_circle_vec( entry_lat, entry_lon,
                                                         G.nodes[node]['y'],
                                                         G.nodes[node]['x'] ) )

    # Linking the lower entry point with the city grid
    node_G = insertionNode( G, entry_lon, entry_lat, entry_low_height )
    # print( node_G )
    delta_z = G.nodes[node_G]['z'] - G.nodes[node_low]['z']
    # print( 'z grid node', G.nodes[node_G]['z'] )
    # print( 'z entry node', G.nodes[node_low]['z'] )

    delta_xy = ox.distance.great_circle_vec( G.nodes[node_G]['y'], G.nodes[node_G]['x'],
                                                 G.nodes[node_low]['y'],
                                                 G.nodes[node_low]['x'] )

    length = ( delta_z ** 2 + delta_xy ** 2 ) ** ( 1 / 2 )

    G.add_edge( node_G, node_low, 0, oneway=False, segment='new', speed=50.0,
                    length=length )
    G.add_edge( node_low, node_G, 0, oneway=False, segment=name, speed=speed,
                    length=length )

    return G, segments

def corridorCreation( G, segments, corridor_coordinates, altitude, speed, capacity, name, config ):
    segments = defineSegment( segments, 0, 0, 0, 0, 0, 0, speed, capacity, name )
    segments[name]['updated'] = False
    segments[name]['new'] = False

    index = 0
    nodes_G = []
    nodes_corridor = []
    for point in corridor_coordinates:
        index += 1
        nodes_corridor += [name + '_' + str( index )]
        point_lon = point[0]
        point_lat = point[1]

                # nodes_G += [insertionNode( G, point_lon, point_lat, altitude )]

        G.addNodeAltitude( nodes_corridor[-1], point_lat, point_lon, altitude )

        # Adds corridor edges to G
        if len( nodes_corridor ) == 1:
            continue

        G.add_edge( nodes_corridor[-2], nodes_corridor[-1], 0, oneway=True, segment=name,
                    speed=speed,
                    length=ox.distance.great_circle_vec( point_lat, point_lon,
                                                         G.nodes[nodes_corridor[-2]]['y'],
                                                         G.nodes[nodes_corridor[-2]]['x'] ) )


    # for entry_node, corridor_point in zip( nodes_G, nodes_corridor ):
    #    delta_z = G.nodes[entry_node]['z'] - G.nodes[corridor_point]['z']
    #    delta_xy = ox.distance.great_circle_vec( G.nodes[entry_node]['y'], G.nodes[entry_node]['x'],
    #                                             G.nodes[corridor_point]['y'],
    #                                             G.nodes[corridor_point]['x'] )

    #    length = ( delta_z ** 2 + delta_xy ** 2 ) ** ( 1 / 2 )

    #    G.add_edge( entry_node, corridor_point, 0, oneway=False, segment='new', speed=50.0,
    #                length=length )
    #    G.add_edge( corridor_point, entry_node, 0, oneway=False, segment=name, speed=speed,
    #                length=length )

        # create entry points
        entryNodes( G, segments, nodes_corridor[-2], name, speed, nodes_corridor[-1], config )

    return G, segments

def str2intList( string ):
    '''
    This function transforms a string containing digits and other characters into a list of integers

    Input:
        string: any string, e.g., '1, 2, 40,,w'
    Output:
        list: list with the integers contained in the string, e.g., ['1', '2', '40']
    '''
    list = []
    for s in string.split():
        if s.isdigit():
             list.append( s )
    return list

def getCorridorCoordinates( corridor, file_path ):
    '''
    This function gets coordinates stored in a txt, csv or excel file

    Input:
        file_path: path to the file storing the coordinates and the corridors associated
        corridor: number of the corridor required
    Output:
        coordinates: a tuple containing lists of [longitude, latitude] for
            the points defining the corridor
    '''
    with open( file_path, 'r' ) as csv_file:
        reader = csv.reader( csv_file, delimiter=';' )
        # fields = reader.next()
        rows = []
        corridor_row = []
        for row in reader:
            rows.append( row )
            if row[0] == corridor:
                corridor_row.append( [float( row[2] ), float( row[1] )] )

    return tuple( corridor_row )


def corridorLoad( G, segments, config ):
    active_corridors = str2intList( config['Corridors']['corridors'] )
    file_path_corridors = config['Corridors']['file_path_corridors']
    altitude = config['Corridors'].getint( 'altitude' )
    delta_z = config['Corridors'].getint( 'delta_z' )
    speed = config['Corridors'].getint( 'speed' )

    for corridor in active_corridors:
        # Creates a unique name for the corridor
        name = 'COR_' + corridor
        name_rev = 'COR_r_'
        # Get corridor coordinates
        corridor_coordinates = getCorridorCoordinates( corridor, file_path_corridors )
        # Creates the segments of the corridor
        G, segments = corridorCreation( G, segments, corridor_coordinates,
                                                      altitude, speed, 50, name, config )
        G, segments = corridorCreation( G, segments, corridor_coordinates[::-1],
                                                      altitude + delta_z, speed, 50, name_rev,
                                                      config )
    return G, segments


if __name__ == '__main__':
    # CONFIG
    config = configparser.ConfigParser()
    config_path = "C:/workspace3/bluesky/nommon/city_model/settings.cfg"
    config.read( config_path )

    corridorLoad( config )
