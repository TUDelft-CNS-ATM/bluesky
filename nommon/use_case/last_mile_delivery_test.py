#!/usr/bin/python

"""
This is a preliminary version of the last mile delivery use case
"""
import configparser
import pickle

from nommon.city_model.city_graph import cityGraph
from nommon.city_model.corridors_implementation import corridorLoad
from nommon.city_model.dynamic_segments import dynamicSegments
from nommon.city_model.multi_di_graph_3D import MultiDiGrpah3D
from nommon.city_model.utils import read_my_graphml


__author__ = 'jbueno'
__copyright__ = '(c) Nommon 2022'

if __name__ == '__main__':
    # -------------- 1. CONFIGURATION FILE -----------------
    """
    This section reads the configuration file.
    Change the config_path to read the desired file
    """
    # CONFIG
    config_path = r"C:\workspace3\bluesky-USEPE-github\nommon\use_case\settings_last_mile_delivery_test.cfg"
    config = configparser.ConfigParser()
    config.read( config_path )

    # -------------- 2. CITY GRAPH -------------------------
    """
    This section creates a city graph or loads the graph defined with the city section of the
    configuration file.
    """
    # City
    if config['City'].getboolean( 'import' ):
        filepath = config['City']['imported_graph_path']
        G = read_my_graphml( filepath )
        G = MultiDiGrpah3D( G )
        # fig, ax = ox.plot_graph( G )
    else:
        G = cityGraph( config )

    # -------------- 3. SEGMENTS ----------------------------
    """
    This section creates a airspace segmentation or loads the segmentation defined with the segment
    section of the configuration file.
    Comment it to neglect the segmentation
    """
    if config['Segments'].getboolean( 'import' ):
        path = config['Segments']['path']
        with open( path, 'rb' ) as f:
            segments = pickle.load( f )
    else:
        segments = None

    G, segments = dynamicSegments( G, config, segments, deleted_segments=None )

    # -------------- 4. CORRIDORS ---------------------------
    """
    This section loads the corridors defined with the corridor section of the configuration file
    Comment it to neglect the creation of corridors
    """
    G, segments = corridorLoad( G, segments, config )
    # G, segments = dynamicSegments( G, config, segments, deleted_segments=None )

