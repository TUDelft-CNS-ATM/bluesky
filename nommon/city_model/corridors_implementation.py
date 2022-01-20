#!/usr/bin/python

"""

"""

from nommon.city_model.dynamic_segments import defineSegment
from nommon.city_model.utils import nearestNode3d
import csv
import json
import math
import os

import osmnx as ox


__author__ = 'jbueno'
__copyright__ = '(c) Nommon 2021'


def entryNodes( G, segments, node, name, speed, next_node, config ):
    """
    This function creates the acceleration lanes for the entry points of the corridors

    Input:
        G - graph
        segments - segments of the graph
        node - corridor node that will be an entrance
        name - for naming the segment ?
        speed - corridor speed
        next_node - next corridor node in the direction of the entrance
        config - configparser.ConfigParser() object that reads the configuration file
            acceleration_lengh - length of the acceleration lane
            n_layers - number of layers of the city grid
            layer_width - width of each layer of the city grid
    Output:
        G
        segments
    """

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
    delta_angle = math.pi / 6  # 30 degrees
    entry_angle = math.atan2( unit_vector_y, unit_vector_x ) + delta_angle

    # Entry point coordinates
    acceleration_lenght = config['Corridors'].getint( 'acceleration_length' )
    entry_x = x + math.cos( entry_angle ) * acceleration_lenght
    entry_y = y + math.sin( entry_angle ) * acceleration_lenght
    entry_lon = math.degrees( entry_x / ( r * math.cos( math.radians( lat ) ) ) )
    entry_lat = math.degrees( entry_y / r )

    # Adding nodes
    node_low = node + '_in_1'
    node_high = node + '_in_2'
    n_layers = config['Layers'].getint( 'number_of_layers' )
    layer_width = config['Layers'].getint( 'layer_width' )
    entry_low_height = n_layers * layer_width
    G.addNodeAltitude( node_low, entry_lat, entry_lon, entry_low_height, name )
    G.addNodeAltitude( node_high, entry_lat, entry_lon, G.nodes[node]['z'], name )

    # Adding edges for those nodes

    # Vertical ascension to the acceleration lane
    G.add_edge( node_low, node_high, 0, oneway=False, segment=name,
                speed=speed, length=G.nodes[node]['z'] - entry_low_height )
    # NOTE: opposite direction - it is entry and exit. TODO: To separate entry and exit points!
    G.add_edge( node_high, node_low, 0, oneway=False, segment=name,
                speed=speed, length=G.nodes[node]['z'] - entry_low_height )

    # Acceleration lane
    G.add_edge( node_high, node, 0, oneway=False, segment=name, speed=speed,
                length=acceleration_lenght )
    # NOTE: opposite direction - it is entry and exit. TODO: To separate entry and exit points!
    G.add_edge( node, node_high, 0, oneway=False, segment=name, speed=speed,
                length=acceleration_lenght )

    # Linking the lower entry point with the city grid
    # Gets closest point in the city grid and distance to it
    node_G = nearestNode3d( G, entry_lon, entry_lat, entry_low_height )

    # Connection to the city grid
    G.add_edge( node_G, node_low, 0, oneway=False, segment='new', speed=50.0,
                length=ox.distance.great_circle_vec( G.nodes[node_G]['y'],
                                                     G.nodes[node_G]['x'],
                                                     G.nodes[node_low]['y'],
                                                     G.nodes[node_low]['x'] ) )

    # NOTE: opposite direction - it is entry and exit. TODO: To separate entry and exit points!
    G.add_edge( node_low, node_G, 0, oneway=False, segment=name, speed=speed,
                length=ox.distance.great_circle_vec( G.nodes[node_G]['y'],
                                                     G.nodes[node_G]['x'],
                                                     G.nodes[node_low]['y'],
                                                     G.nodes[node_low]['x'] ) )

    return G, segments


def corridorCreation( G, segments, corridor_coordinates, altitude, speed, capacity, name, config ):
    '''
    This function creates a corridor as defined by its coordinates
    and adds entry points for the corridor

    Input:
        G - graph
        segments - graph segments
        corridor_coordinates - coordinates that define the shape of the corridor
        altitude - altitude of the corridor
        speed - speed defined for the corridor
        capacity - capacity of the corridor
        name - name of the corridor
        config - configparser.ConfigParser() object that reads the configuration file

    Output:
        G - graph updated with the corridors
        segments - segments updated with the corridors
    '''

    segments = defineSegment( segments, 0, 0, 0, 0, 0, 0, speed, capacity, name )
    segments[name]['updated'] = False
    segments[name]['new'] = False

    index = 0
    # nodes_G = []
    nodes_corridor = []
    for point in corridor_coordinates:
        index += 1
        nodes_corridor += [name + '_' + str( index )]
        point_lon = point[0]
        point_lat = point[1]

        # nodes_G += [insertionNode( G, point_lon, point_lat, altitude )]

        G.addNodeAltitude( nodes_corridor[-1], point_lat, point_lon, altitude, name )
        # G.add_node( nodes_corridor[-1], y=point_lat, x=point_lon, z=altitude, segment=name )

        # Adds corridor edges to G
        if len( nodes_corridor ) == 1:
            continue

        G.add_edge( nodes_corridor[-2], nodes_corridor[-1], 0, oneway=True, segment=name,
                    speed=speed,
                    length=ox.distance.great_circle_vec( point_lat, point_lon,
                                                         G.nodes[nodes_corridor[-2]]['y'],
                                                         G.nodes[nodes_corridor[-2]]['x'] ) )

        # create entry points
        entryNodes( G, segments, nodes_corridor[-2], name, speed, nodes_corridor[-1], config )

    # Checks the distance between start and end points of the corridor.
    # If they are less than delta, it considers the corridor as circular and joins both nodes
    od_length = ox.distance.great_circle_vec( G.nodes[nodes_corridor[0]]['y'],
                                              G.nodes[nodes_corridor[0]]['x'],
                                              G.nodes[nodes_corridor[-1]]['y'],
                                              G.nodes[nodes_corridor[-1]]['x'] )
    delta = 25.0  # tolerance distance (m) to consider the corridor as a closed path
    if od_length < delta:
        G.add_edge( nodes_corridor[-1], nodes_corridor[0], 0, oneway=True, segment=name,
                    speed=speed,
                    length=ox.distance.great_circle_vec( G.nodes[nodes_corridor[0]]['y'],
                                                         G.nodes[nodes_corridor[0]]['x'],
                                                         G.nodes[nodes_corridor[-1]]['y'],
                                                         G.nodes[nodes_corridor[-1]]['x'] ) )
    return G, segments


def str2intList( string ):
    '''
    This function transforms a string containing digits and other characters into a list of integers

    Input:
        string: any string, e.g., '1, 2, 40,,w'
    Output:
        list: list with the integers contained in the string, e.g., ['1', '2', '40']
    '''
    int_list = []
    for s in string.split():
        if s.isdigit():
            int_list.append( s )
    return int_list


def getCorridorCoordinates( corridor, file_path ):
    '''
    This function gets coordinates stored in a json or csv file

    Input:
        file_path: path to the file storing the coordinates and the corridors associated
        corridor: number (csv) or name (geojson) of the corridor required
    Output:
        coordinates: a tuple containing lists of [longitude, latitude] for
            the points defining the corridor
    '''
    _, file_extension = os.path.splitext( file_path )
    if file_extension == ".csv":
        with open( file_path, 'r' ) as csv_file:
            reader = csv.DictReader( csv_file )
            rows = []
            corridor_row = []
            for row in reader:
                rows.append( row )
                if row['corridor'] == corridor:
                    corridor_row.append( [float( row['lon'] ), float( row['lat'] )] )

    elif file_extension == ".geojson":
        with open( file_path ) as json_file:
            gj_in = json.load( json_file )
            corridor_row = []
            for feature in gj_in["features"]:
                corr_id = feature['properties']['id']
                if str( corr_id ) == corridor:
                    geo = feature["geometry"]
                    for point in geo["coordinates"]:
                        corridor_row.append( [point[0], point[1]] )

    return tuple( corridor_row )


def corridorLoad( G, segments, config ):
    '''
    This function reads the parameters from the configuration file and executes the corridor
    creation function for all the active corridors defined in the configuration file

    Input:
        G - graph
        segments
        config - configparser.ConfigParser() object that reads the configuration file

    Outputs:
        G
        segments
    '''
    # reads the list of active corridors defined in the settings file
    active_corridors = str2intList( config['Corridors']['corridors'] )
    print( 'Active corridors', active_corridors )
    # reads the path to the csv file containing the points of the corridors
    file_path_corridors = config['Corridors']['file_path_corridors']
    # Reads the altitude defined for the corridors
    altitude = config['Corridors'].getint( 'altitude' )
    # Reads the altitude gap between a corridor and the one above it with opposite direction
    delta_z = config['Corridors'].getint( 'delta_z' )
    # Reads the speed of the corridors
    #  so far, it is the same for all of them
    speed = config['Corridors'].getint( 'speed' )

    for corridor in active_corridors:
        # Creates a unique name for the corridor
        name = 'COR' + corridor
        # Creates a name for the a corridor in the opposite direction
        name_rev = 'COR' + corridor + 'r'
        # Get corridor coordinates
        corridor_coordinates = getCorridorCoordinates( corridor, file_path_corridors )
        # Check if the corridor exists
        if corridor_coordinates == ():
            print( 'Corridor number {0} is not contained in {1}'.format( corridor, file_path_corridors ) )
            continue
        # Creates the segments of the corridor
        G, segments = corridorCreation( G, segments, corridor_coordinates,
                                        altitude, speed, 50, name, config )
        G, segments = corridorCreation( G, segments, corridor_coordinates[::-1],
                                        altitude + delta_z, speed, 50, name_rev, config )
    return G, segments


if __name__ == '__main__':
    file_path = "C:/workspace3/bluesky_organisation_clone/nommon/city_model/data/usepe-hannover-corridors.csv"
    corridor = "1"
    corridor_coord = getCorridorCoordinates( corridor, file_path )
    print( corridor_coord )

    if corridor_coord == ():
        print( 'Corridor number {0} is not contained in {1}'.format( corridor, file_path ) )
