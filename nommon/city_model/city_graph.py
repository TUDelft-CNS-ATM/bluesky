#!/usr/bin/python

"""
Script to create the graph of a city
"""
import configparser
import string

from auxiliar import read_my_graphml
from building_height import readCity
from city_structure import mainSectorsLimit
from multi_di_graph_3D import MultiDiGrpah3D
import networkx as nx
import osmnx as ox


__author__ = 'jbueno'
__copyright__ = '(c) Nommon 2021'




def cityGraph( config ):
#     config = configparser.ConfigParser()
#     config.read( config_path )

    # We import a graph of the city with all the streets = edges, and intersections = nodes

    # Study zone
#     hannover = ( config['City'].getfloat( 'hannover_lat' ),
#                  config['City'].getfloat( 'hannover_lon' ) )  # Hannover coordinates

    hannover = ( config['City'].getfloat( 'hannover_lat_min' ),
                 config['City'].getfloat( 'hannover_lat_max' ),
                 config['City'].getfloat( 'hannover_lon_min' ),
                 config['City'].getfloat( 'hannover_lon_max' ) )  # Hannover coordinates

#     zone_size = config['City'].getint( 'zone_size' )  # meters

#     G = ox.graph_from_point( hannover, dist=zone_size, network_type="drive", simplify=False )
    print( 'Obtaining the graph from OSM...' )
    G = ox.graph_from_bbox( hannover[1], hannover[0], hannover[3], hannover[2],
                            network_type="drive", simplify=False )
    G = MultiDiGrpah3D( G )  # We create a graph with our clase MultiDiGrpah3D

    # We save the nodes
    nodes_to_be_removed = list( G.nodes )

    # An attribute "altitude" = 0 is defined for each node
    G.defGroundAltitude()

    # We create some layers
    letters = list( string.ascii_uppercase )
    layers = letters[0:config['Layers'].getint( 'number_of_layers' )]

    G.addLayer( 'A', config['Layers'].getint( 'layer_width' ) )
    G.remove_nodes_from( nodes_to_be_removed )

    # Simplify the graph
    if config['Options'].getboolean( 'simplify' ):
        G.simplifyGraph( config )

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

    G.addDiagonalEdges( sectors, config )

    if config['Options'].getboolean( 'one_way' ):
        G.defOneWay( config )

    # We plot the graph
#     fig, ax = ox.plot_graph( G )
    print( 'Saving the graph...' )

    filepath = r"C:\workspace3\USEPE-CityEnvironment\src\data\hannover.graphml"
    ox.save_graphml( G, filepath )
    return G

if __name__ == '__main__':
    config_path = "C:/workspace3/USEPE-CityEnvironment/src/settings.cfg"
    G = cityGraph( config_path )
    G = ox.speed.add_edge_speeds( G, fallback=10 )
    G = ox.speed.add_edge_travel_times( G )
    edges = ox.graph_to_gdfs( G, nodes=False )
    edges["segment"] = edges["segment"].astype( str )
    print( edges.groupby( "segment" )[["length", "speed_kph", "travel_time"]].mean().round( 1 ) )
    print( edges.groupby( "segment" ) )
