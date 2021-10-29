#!/usr/bin/python

"""
Additional functions
"""
import string

from osmnx.io import _convert_node_attr_types, _convert_bool_string, _convert_edge_attr_types

from multi_di_graph_3D import MultiDiGrpah3D
import networkx as nx


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


if __name__ == '__main__':
    pass
