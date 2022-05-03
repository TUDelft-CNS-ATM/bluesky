#!/usr/bin/python

"""
Transform the segments to the format used by city_module.py
"""
import json
import pickle
import random

import geopandas as gpd


__author__ = 'jbueno'
__copyright__ = '(c) Nommon 2022'


def defineSegment( segments, lon_min, lon_max, lat_min, lat_max, z_min, z_max,
                   speed, capacity, name, color ):
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
            color (string): segment class

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
                      'updated': True,
                      'class': color}
    return segments


def offlineSegments( rules_path, segment_path ):
    f = open( rules_path )
    rules = json.load( f )

    cells = gpd.read_file( segment_path, driver="GeoJSON" )

    segments = {}
    for index, row in cells.iterrows():
        lon_min = min( row["geometry"].boundary.coords.xy[0] )
        lon_max = max( row["geometry"].boundary.coords.xy[0] )
        lat_min = min( row["geometry"].boundary.coords.xy[1] )
        lat_max = max( row["geometry"].boundary.coords.xy[1] )
        z_min = row["floor"]
        z_max = row["ceiling"]
        name = str( index )
        speed = float( random.randint( 5, 15 ) )
        capacity = random.randint( 1, 20 )
        new = True
        updated = True
        if row["class"] == "black":
            speed = 0
            capacity = 0
            defineSegment( segments, lon_min, lon_max, lat_min, lat_max, z_min, z_max,
                           speed, capacity, name, 'black' )
        elif row["class"] == "red":
            speed = 0
            capacity = 0
            defineSegment( segments, lon_min, lon_max, lat_min, lat_max, z_min, z_max,
                           speed, capacity, name, 'red' )
        elif row["class"] == "yellow":
            speed = float( random.randint( rules['classes']['yellow']['velocity'][0], rules['classes']['yellow']['velocity'][1] ) )
            capacity = random.randint( 1, 20 * rules['classes']['yellow']['capacity_factor'] )
            defineSegment( segments, lon_min, lon_max, lat_min, lat_max, z_min, rules['classes']['yellow']['altitude'][0],
                           0, 0, name + '_ground', 'yellow' )
            defineSegment( segments, lon_min, lon_max, lat_min, lat_max, rules['classes']['yellow']['altitude'][0], z_max,
                           speed, capacity, name, 'yellow' )
        elif row["class"] == "green":
            speed = float( random.randint( rules['classes']['green']['velocity'][0], rules['classes']['green']['velocity'][1] ) )
            capacity = random.randint( 1, 20 * rules['classes']['green']['capacity_factor'] )
            defineSegment( segments, lon_min, lon_max, lat_min, lat_max, z_min, rules['classes']['green']['altitude'][0],
                           0, 0, name + '_ground', 'green' )
            defineSegment( segments, lon_min, lon_max, lat_min, lat_max, rules['classes']['green']['altitude'][0], z_max,
                           speed, capacity, name, 'green' )
        elif row["class"] == "grey":
            speed = float( random.randint( rules['classes']['grey']['velocity'][0], rules['classes']['grey']['velocity'][1] ) )
            capacity = random.randint( 1, 20 )
            defineSegment( segments, lon_min, lon_max, lat_min, lat_max, z_min, rules['classes']['grey']['altitude'][1],
                           speed, capacity, name, 'grey' )
            defineSegment( segments, lon_min, lon_max, lat_min, lat_max, rules['classes']['grey']['altitude'][1], z_max,
                           0, 0, name + 'ceiling', 'grey' )
        elif row["class"] == "white":
            speed = float( random.randint( 5, 50 ) )
            capacity = random.randint( 1, 20 )
            defineSegment( segments, lon_min, lon_max, lat_min, lat_max, z_min, z_max,
                           speed, capacity, name, 'white' )

    with open( 'offline_segments.pkl', 'wb' ) as f:
        pickle.dump( segments, f )

    with open( 'offline_segments.pkl', 'rb' ) as f:
        loaded_dict = pickle.load( f )


def referenceSegments( rules_path, segment_path ):
    f = open( rules_path )
    rules = json.load( f )

    cells = gpd.read_file( segment_path, driver="GeoJSON" )

    segments = {}
    for index, row in cells.iterrows():
        lon_min = min( row["geometry"].boundary.coords.xy[0] )
        lon_max = max( row["geometry"].boundary.coords.xy[0] )
        lat_min = min( row["geometry"].boundary.coords.xy[1] )
        lat_max = max( row["geometry"].boundary.coords.xy[1] )
        z_min = row["floor"]
        z_max = row["ceiling"]
        name = str( index )
        # speed = float( random.randint( 5, 15 ) )
        # capacity = random.randint( 1, 20 )

        speed = 20
        capacity = 999

        new = True
        updated = True
        if row["class"] == "black":
            speed = 0
            capacity = 0
            defineSegment( segments, lon_min, lon_max, lat_min, lat_max, z_min, z_max,
                           speed, capacity, name, 'black' )
        else:
            # speed = float( random.randint( 5, 50 ) )
            # capacity = random.randint( 1, 20 )
            defineSegment( segments, lon_min, lon_max, lat_min, lat_max, z_min, z_max,
                           speed, capacity, name, 'white' )

    with open( 'reference_segments.pkl', 'wb' ) as f:
        pickle.dump( segments, f )

    with open( 'reference_segments.pkl', 'rb' ) as f:
        loaded_dict = pickle.load( f )


if __name__ == '__main__':
    # ----------------- Defined by user ------------------ #
    rules_path = r'../config/rules.json'
    segment_path = r"../data/examples/hannover.geojson"

    # ----------------------------------------------------- #
    # offlineSegments(rules_path, segment_path)
    referenceSegments( rules_path, segment_path )

