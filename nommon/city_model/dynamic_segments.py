#!/usr/bin/python

"""

"""

__author__ = 'jbueno'
__copyright__ = '(c) Nommon 2021'

import random
import time

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd


def defineSegment( segments, lon_min, lon_max, lat_min, lat_max, z_min, z_max,
                   speed, capacity, name ):
    """
    Define a new segment. The segment is consider as a prism.

    Parameter "new" : boolean variable. If "True" the segment is new; "False" otherwise
    Parameter "updated : boolean variable. If "True" some segment property is modified (typically
                        the speed will be updated); "False" otherwise

    Args:
            segments (dictionary): dictionary with the segment information
            lon_min (float): minimum longitude
            lon_max (float): maximum longitude
            lat_min (float): minimum latitude
            lat_max (float): maximum latitude
            z_min (float): minimum altitude
            z_max (float): maximum altitude
            speed (float): maximum speed of the segment
            capacity (integer): capacity of the segment in terms of number of drones
            name (string): id of the segment

    Returns:
            segments (dictionary): updated dictionary with the segment information
    """
    segments[name] = {'lon_min': lon_min,
                      'lon_max': lon_max,
                      'lat_min': lat_min,
                      'lat_max': lat_max,
                      'z_min': z_min,
                      'z_max': z_max,
                      'speed': speed,
                      'capacity': capacity,
                      'new': True,
                      'updated': True}
    return segments


def divideAirspaceSegments( lon_min, lon_max, lat_min, lat_max, z_min, z_max, divisions_lon,
                            divisions_lat, divisions_z ):
    """
    Create a default segment division of the airspace. The default segmentation divide the airspace
    evenly

    Args:
            lon_min (float): float representing the minimum longitude of the graph
            lon_max (float): float representing the maximum longitude of the graph
            lat_min (float): float representing the minimum latitude of the graph
            lat_max (float): float representing the maximum latitude of the graph
            z_min (float): float representing the minimum altitude of the graph
            z_max (float): float representing the minimum altitude of the graph
            divisions_lon (integer): integer indicating the number of segments in the longitude direction
            divisions_lat (integer): integer indicating the number of segments in the latitude direction
            divisions_z (integer): integer indicating the number of segments in the z direction

    Returns:
            segments (dictionary): dictionary with all the information about segments

    """

    print( 'Creating segments...' )
    delta_lon = ( lon_max - lon_min ) / divisions_lon
    delta_lat = ( lat_max - lat_min ) / divisions_lat
    delta_z = ( z_max - z_min ) / divisions_z

    segments = {}
    for i in range( divisions_lon ):
        for j in range( divisions_lat ):
            for k in range( divisions_z ):
                name = 'segment_' + str( i ) + '_' + str( j ) + '_' + str( k )
                lon_min_seg = lon_min + i * delta_lon
                lon_max_seg = lon_min + ( i + 1 ) * delta_lon
                lat_min_seg = lat_min + j * delta_lat
                lat_max_seg = lat_min + ( j + 1 ) * delta_lat
                z_min_seg = z_min + k * delta_z
                z_max_seg = z_min + ( k + 1 ) * delta_z
                speed_seg = float( random.randint( 5, 15 ) )
                capacity_seg = random.randint( 1, 20 )

                segments = defineSegment( segments, lon_min_seg, lon_max_seg, lat_min_seg,
                                          lat_max_seg, z_min_seg, z_max_seg, speed_seg,
                                          capacity_seg, name )

    # We create a segment to be used in case we have an edge not included in this division
    segments = defineSegment( segments, 0, 0, 0,
                              0, 0, 0, 0.0,
                              0, 'N/A' )

    return segments


def selectNodesWithNewSegments( G, segments, deleted_segments ):
    """
    Select the nodes affected by the new segmentation. It takes the new nodes, the nodes belonging
    to new segments and the nodes belonging to segments that has been deleted

    Args:
            G (graph): graph representing the city
            segments_df (DataFrame): dataframe with the segment information
            deleted_segments (list): a list containing the segments that has been deleted

    """
    new_segments = list( segments.index )
    nodes = ox.graph_to_gdfs( G, edges=False, node_geometry=False )

    # When a node is created for first time, the segment parameter is "new". So, the condition
    # includes all the new nodes
    cond = nodes['segment'] == 'new'
    # Nodes belonging to new segments
    cond = cond | ( nodes['segment'].isin( new_segments ) )
    # Nodes belonging to deleted segments
    if deleted_segments is not None:
        cond = cond | ( nodes['segment'].isin( deleted_segments ) )

    df = nodes[cond]
    return df.index


def assignSegmet2Edge( G, segments_df, deleted_segments ):
    """
    Assign to each node and edge the segment it belongs to. If the origin node of an edge belongs to
    the segment, then the edge belongs to the segment.

    Args:
            G (graph): graph representing the city
            segments_df (DataFrame): dataframe with the segment information
            deleted_segments (list): a list containing the segments that has been deleted

    Returns:
            G (graph): updated graph
    """

    if segments_df.empty:
        print( 'No new segments' )
        return G

    print( 'Assigning segments...' )
    # We select the nodes belonging to the new segments
    nodes_affected = selectNodesWithNewSegments( G, segments_df, deleted_segments )
    for node in nodes_affected:
        if node[0:2] == 'COR':
            # corridors do not depend on this segmentation
            continue

        node_lon = G.nodes[node]['x']
        node_lat = G.nodes[node]['y']
        node_z = G.nodes[node]['z']

        # We check which is the segment associated to the node
        cond = ( segments_df['lon_min'] <= node_lon ) & ( segments_df['lon_max'] > node_lon ) & \
            ( segments_df['lat_min'] <= node_lat ) & ( segments_df['lat_max'] > node_lat ) & \
            ( segments_df['z_min'] <= node_z ) & ( segments_df['z_max'] > node_z )

        if segments_df[cond].empty:
            segment_name = 'N/A'
        else:
            segment_name = segments_df[cond].index[0]

        G.nodes[node]['segment'] = segment_name
        connected_edges = list( G.neighbors( node ) )
        for edge in connected_edges:
            # If the origin node of an edge belongs to the segment, then the edge belongs to the
            # segment.
            G.edges[node, edge, 0 ]['segment'] = segment_name

    return G


def updateSegmentVelocity( G, segments ):
    """
    Update the edge speed according to the new segmentation.

    Args:
            G (graph): graph representing the city
            segments (DataFrame): dataframe with the segment information
    Returns:
            G (graph): updated graph
    """
    if segments.empty:
        print( 'No new velocities' )
        return G
    print( 'Updating segment velocity...' )

    updated_segments = list( segments.index )

    edges = ox.utils_graph.graph_to_gdfs( G, nodes=False, fill_edge_geometry=False )

    cond = edges['segment'] == 'N/A'

    cond = cond | ( edges['segment'].isin( updated_segments ) )

    pd.set_option( 'mode.chained_assignment', None )
    edges['speed'][cond] = edges[cond]['segment'].apply( lambda segment_name:
                                                         segments.loc[segment_name]['speed'] )

    nx.set_edge_attributes( G, values=edges["speed"], name="speed" )
    return G


def addTravelTimes( G, precision=4 ):
    """
    Calculate the travel time of all the edges.
    Args:
            G (graph): graph representing the city
            precision (integer): integer to round the travel time
    Returns:
            G (graph): updated graph
    """
    print( 'Updating travel times...' )
    edges = ox.utils_graph.graph_to_gdfs( G, nodes=False, fill_edge_geometry=False )

    # verify edge length and speed_kph attributes exist and contain no nulls
    if not ( "length" in edges.columns and "speed" in edges.columns ):
        raise KeyError( "all edges must have `length` and `speed` attributes." )
    else:
        if ( pd.isnull( edges["length"] ).any() or pd.isnull( edges["speed"] ).any() ):
            raise ValueError( "edge `length` and `speed_kph` values must be non-null." )

    # convert distance meters to km, and speed km per hour to km per second
    distance_km = edges["length"] / 1000
    speed_km_sec = edges["speed"] / ( 60 * 60 )

    # calculate edge travel time in seconds
    travel_time = distance_km / speed_km_sec

    # replace the infinity values by a high number
    travel_time.replace( [np.inf, -np.inf], 9999999999, inplace=True )

    # add travel time attribute to graph edges
    edges["travel_time"] = travel_time.round( precision ).values
    nx.set_edge_attributes( G, values=edges["travel_time"], name="travel_time" )

    return G


def dynamicSegments( G, config, segments=None, deleted_segments=None ):
    """
    Assign segments to edges, and update velocities and and travel times. If segments information is
    not provided, a default segmentation is created.

    Args:
            G (graph): graph representing the city
            config (configuration file): configuration file with all the relevant information
            segments (dictionary): dictionary with all the information about segments
            deleted_segments (list): a list containing the segments that has been deleted

    Returns:
            G (graph): updated graph representing the city according to the segmetns
            segments (dictionary): updated dictionary with all the information about segments
    """
    print( 'Updating segments...' )
    if not segments:
        segments = divideAirspaceSegments( config['City'].getfloat( 'hannover_lon_min' ),
                                           config['City'].getfloat( 'hannover_lon_max' ),
                                           config['City'].getfloat( 'hannover_lat_min' ),
                                           config['City'].getfloat( 'hannover_lat_max' ),
                                           0,
                                           config['Layers'].getfloat( 'layer_width' ) *
                                           ( config['Layers'].getfloat( 'number_of_layers' ) + 1 ),
                                           4, 4, 2 )

    segments_df = pd.DataFrame.from_dict( segments, orient='index' )

    # We select only the new segments. The assignment of segments to edges is performed only in
    # these segments
    new_segments = segments_df[segments_df['new'] == True ]

    # Assign segments
    G = assignSegmet2Edge( G, new_segments, deleted_segments )

    # We select only the updated segments. The speed update of segments is performed only in
    # these segments
    updated_segments = segments_df[segments_df['updated'] == True ]

    # Update segment velocity
    G = updateSegmentVelocity( G, updated_segments )

    # Add travel times to the graph
    G = addTravelTimes( G )

    segments_df['new'] = False
    segments_df['updated'] = False

    segments = segments_df.to_dict( orient='index' )
    print( 'Dynamic segments completed' )
    return G, segments


if __name__ == '__main__':
    pass
    # filepath = "./data/hannover.graphml"
    #
    # from utils import read_my_graphml
    # from multi_di_graph_3D import MultiDiGrpah3D
    #
    # G = read_my_graphml( filepath )
    # G = MultiDiGrpah3D( G )
    # segments = divideAirspaceSegments( 0, 20, 50, 54, 0, 250, 4, 4, 2 )
    # G = assignSegmet2Edge( G, segments )
    # G = updateSegmentVelocity( G, segments )
    # edges = ox.graph_to_gdfs( G, nodes=False )
    # print( edges['segment'] )
    # print( edges.columns )
    # print( edges['speed'] )
    # print( segments['segment_1_2_0'] )
    # G = addTravelTimes( G )
    #
    # from nommon.city_model.path_planning import trajectoryCalculation
    # orig = ( 9.74 , 52.36 )
    # dest = ( 9.78 , 53.38 )
    # print( trajectoryCalculation( G, orig, dest ) )



