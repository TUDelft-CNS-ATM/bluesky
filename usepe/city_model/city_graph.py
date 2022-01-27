#!/usr/bin/python

"""
Script to create the graph of a city
"""
import configparser
import os
import string

from usepe.city_model.building_height import readCity
from usepe.city_model.city_structure import mainSectorsLimit
from usepe.city_model.multi_di_graph_3D import MultiDiGrpah3D
from usepe.city_model.utils import read_my_graphml
import networkx as nx
import osmnx as ox


__author__ = 'jbueno'
__copyright__ = '(c) Nommon 2021'


def cityGraph( config ):
    """
    It creates a 3D graph of a city based on the parameters of the configuration file

    Args:
            config (configuration file): A configuration file with all the relevant information

    Returns:
            G (graph): graph representing the available urban airspace for drones
    """

    # We import a graph of the city with all the streets = edges, and intersections = nodes

    # Study zone + import the graph from OSM
    if config['City']['mode'] == 'square':
        hannover = ( config['City'].getfloat( 'hannover_lat' ),
                     config['City'].getfloat( 'hannover_lon' ) )  # Hannover coordinates
        zone_size = config['City'].getint( 'zone_size' )  # meters

        print( 'Obtaining the graph from OSM...' )
        G = ox.graph_from_point( hannover, dist=zone_size, network_type="drive", simplify=False )
    else:
        hannover = ( config['City'].getfloat( 'hannover_lat_min' ),
                     config['City'].getfloat( 'hannover_lat_max' ),
                     config['City'].getfloat( 'hannover_lon_min' ),
                     config['City'].getfloat( 'hannover_lon_max' ) )  # Hannover coordinates

        print( 'Obtaining the graph from OSM...' )
        G = ox.graph_from_bbox( hannover[1], hannover[0], hannover[3], hannover[2],
                                network_type="drive", simplify=False )

    # We create a graph with our class MultiDiGrpah3D
    G = MultiDiGrpah3D( G )

    fig, ax = ox.plot_graph( G )

    # We save the nodes of the streets
    nodes_to_be_removed = list( G.nodes )

    # An attribute "altitude" = 0 is defined for each node
    G.defGroundAltitude()

    # We create the first layers
    letters = list( string.ascii_uppercase )
    layers = letters[0:config['Layers'].getint( 'number_of_layers' )]

    G.addLayer( 'A', config['Layers'].getint( 'layer_width' ) )
    G.remove_nodes_from( nodes_to_be_removed )

    # Simplify the graph
    if config['Options'].getboolean( 'simplify' ):
        G.simplifyGraph( config )

    # Create the rest of layers
    for elem in layers[1:]:
        G.addLayer( elem, config['Layers'].getint( 'layer_width' ) )

    # Building data
    directory_buildings = config['BuildingData']['directory_hannover']
    building_dict = readCity( directory_buildings )

    # Sectors are defined
    lon_min = config['BuildingData'].getfloat( 'lon_min' )
    lon_max = config['BuildingData'].getfloat( 'lon_max' )
    lat_min = config['BuildingData'].getfloat( 'lat_min' )
    lat_max = config['BuildingData'].getfloat( 'lat_max' )
    divisions = config['BuildingData'].getint( 'divisions' )

    sectors, building_dict = mainSectorsLimit( lon_min, lon_max, lat_min, lat_max, divisions,
                                               building_dict )

    # We allow drone movement above buildings
    G.addDiagonalEdges( sectors, config )

    if config['Options'].getboolean( 'one_way' ):  # if we want a one way graph
        G.defOneWay( config )

    # We plot the graph
    fig, ax = ox.plot_graph( G )

    print( 'Saving the graph...' )
    filepath = config['Outputs']['graph_path']
    ox.save_graphml( G, filepath )
    return G


if __name__ == '__main__':
    pass
    # config_path = "C:/workspace3/bluesky/nommon/city_model/settings.cfg"
    # print( 'config path correct?', os.path.isfile( config_path ) )
    # G = cityGraph( config_path )
    # G = ox.speed.add_edge_speeds( G, fallback=10 )
    # G = ox.speed.add_edge_travel_times( G )
    # edges = ox.graph_to_gdfs( G, nodes=False )
    # edges["segment"] = edges["segment"].astype( str )
    # print( edges.groupby( "segment" )[["length", "speed_kph", "travel_time"]].mean().round( 1 ) )
    # print( edges.groupby( "segment" ) )
