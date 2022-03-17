#!/usr/bin/python

"""
Additional functions
"""
import string

from osmnx.io import _convert_node_attr_types, _convert_bool_string, _convert_edge_attr_types

from usepe.city_model.multi_di_graph_3D import MultiDiGrpah3D
import networkx as nx
import osmnx as ox


__author__ = 'jbueno'
__copyright__ = '(c) Nommon 2021'


def read_my_graphml( filepath ):
    """
    Read a previously computed graph
    Args:
            filepath (string): string representing the path where the graph is stored
    Returns:
            G (graph): graph stored at the filepath
    """

    default_node_dtypes = {
        "elevation": float,
        "elevation_res": float,
        "lat": float,
        "lon": float,
        "street_count": int,
        "x": float,
        "y": float,
        "z": float,
    }
    default_edge_dtypes = {
        "bearing": float,
        "grade": float,
        "grade_abs": float,
        "length": float,
        "oneway": _convert_bool_string,
        "osmid": int,
        "speed": float,
        "travel_time": float,
        "maxspeed": float,
    }

    # read the graphml file from disk
    print( 'Reading the graph...' )
    G = nx.read_graphml( filepath, force_multigraph=True )

    # convert graph/node/edge attribute data types
    G.graph.pop( "node_default", None )
    G.graph.pop( "edge_default", None )
    G = _convert_node_attr_types( G, default_node_dtypes )
    G = _convert_edge_attr_types( G, default_edge_dtypes )
    G = MultiDiGrpah3D( G )

    return G


def layersDict( config ):
    """
    Create a dictionary with the information about the altitude of each layer
    Args:
            config (configuration file): A configuration file with all the relevant information
    Returns:
            layers_dict (dict): dictionary with keys=layers values=altitude [m]
    """
    letters = list( string.ascii_uppercase )
    total_layers = letters[0:config['Layers'].getint( 'number_of_layers' )]
    layer_width = config['Layers'].getint( 'layer_width' )
    altitude = 0
    layers_dict = {}
    for layer in total_layers:
        altitude += layer_width
        layers_dict[layer] = altitude

    return layers_dict


def nearestNode3d( G, lon, lat, altitude ):
    '''
    This function gets the closest node of the city graph nodes with respect
    to a given reference point (lat, lon, alt)

    Input:
        G - graph
        lon - longitude of the reference point
        lat - latitude of the reference point
        altitude - altitude of the reference point

    Output:
        nearest_node - closest node of the city graph nodes with respect to the reference point
        distance - distance between the nearest node and the reference point (lat, lon, alt)
    '''
    # The nodes are filtered to exclude corridor nodes
    nodes = list( G.nodes )
    filtered_latlon = list( filter( lambda node: str( node )[:3] != 'COR', nodes ) )
    # Iterates to get the closest one
    nearest_node = filtered_latlon[0]
    delta_xyz = ( ( G.nodes[nearest_node]['z'] - altitude ) ** 2 +
                  ( G.nodes[nearest_node]['y'] - lat ) ** 2 +
                  ( G.nodes[nearest_node]['x'] - lon ) ** 2 )

    for node in filtered_latlon[1:]:
        delta_xyz_aux = ( ( G.nodes[node]['z'] - altitude ) ** 2 +
                          ( G.nodes[node]['y'] - lat ) ** 2 +
                          ( G.nodes[node]['x'] - lon ) ** 2 )
        if delta_xyz_aux < delta_xyz:
            delta_xyz = delta_xyz_aux
            nearest_node = node
    return nearest_node

def checkIfNoFlyZone( lat, lon, alt, G, segments ):
    '''
    This function checks if the point or its nearest node is within a no-fly zone
    '''
    # Get closed segments
    closed_segments = {}
    for segment_id, segment in segments.items():
        if segment['speed'] == 0:
            closed_segments[segment_id] = segment
    # Check if the point is inside a no-fly zone
    for segment_id, segment in closed_segments.items():
        # Origin
        if lat > segment['lat_min'] and  lat < segment['lat_max']:
            if lon > segment['lon_min'] and  lon < segment['lon_max']:
                print( 'Point in no fly zone: lat {0}, lon {1}'.format( lat, lon ) )
                return True
    # Check if the closest node of the graph is inside a no-fly zone
    if alt == None:
        nearest_node = ox.distance.nearest_nodes( G, X=lon, Y=lat )
    else:
        nearest_node = nearestNode3d( G, lon, lat, alt )
    speed = segments[G.nodes[nearest_node]['segment']]['speed']
    cap = segments[G.nodes[nearest_node]['segment']]['capacity']
    if speed == 0:
        return True
    return False

if __name__ == '__main__':
    pass
