#!/usr/bin/python

"""

"""
import configparser
import time

from nommon.city_model.auxiliar import read_my_graphml
from nommon.city_model.city_graph import cityGraph
from nommon.city_model.corridors_implementation import corridorCreation
from nommon.city_model.dynamic_segments import dynamicSegments
from nommon.city_model.multi_di_graph_3D import MultiDiGrpah3D
from nommon.city_model.path_planning import trajectoryCalculation, printRoute
import numpy as np
import osmnx as ox


__author__ = 'jbueno'
__copyright__ = '(c) Nommon 2021'


if __name__ == '__main__':

    # CONFIG
    config = configparser.ConfigParser()
    config_path = "C:/workspace3/bluesky/nommon/city_model/settings.cfg"
    config.read( config_path )

    # City
    if config['City'].getboolean( 'import' ):
        filepath = r"C:\workspace3\bluesky\nommon\city_model\data\hannover_segments.graphml"
        G = read_my_graphml( filepath )
        G = MultiDiGrpah3D( G )
#         fig, ax = ox.plot_graph( G )
        segments = np.load( 'my_segments.npy', allow_pickle='TRUE' ).item()
    else:
        G = cityGraph( config )

    # Segments
#     G, segments = dynamicSegments( G, config, segments=None )
#
#     segments[list( segments.keys() )[0]]['new'] = True
#     segments[list( segments.keys() )[0]]['updated'] = True
#     G, segments = dynamicSegments( G, config, segments )
#
#     print( 'Saving the graph...' )
#     filepath = "./data/hannover_segments.graphml"
#     ox.save_graphml( G, filepath )
#
#     np.save( 'my_segments.npy', segments )

    # Corridors
    G, segments = corridorCreation( G, segments, ( [9.73, 52.375], [9.77, 52.375 ] ),
                                    200, 100, 50, 'COR_1' )

    # Route
    orig = [9.73, 52.38]
    dest = [9.77, 52.38 ]
    route = trajectoryCalculation( G, orig, dest )
    printRoute( G, route )

    print( 'Finish.' )
