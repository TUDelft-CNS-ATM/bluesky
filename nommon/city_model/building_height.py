#!/usr/bin/python

"""
A module for importing and processing the building heights from gml file
"""
import os
import statistics

from pyproj import Transformer

import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET


__author__ = 'jbueno'
__copyright__ = '(c) Nommon 2021'


def gmlFiles( directory ):
    """
    Create a list with all the paths for the gml files

    Args:
             directory (string): directory where the gml files are stored
    Returns:
            gml_files (list): list with all the paths for the gml files
    """
    gml_files = []
    for file in os.listdir( directory ):
        if file.endswith( ".gml" ):
            gml_files += [directory + '\\' + file]

    return gml_files


def footprintListOfLists( lst ):
    """
    Create a list with the footprint information. Each element is a list with x and y coordinates
    of the building footprint

    Args:
            lst (list): list with the coordinates

    Returns:
            footprint (list): list of lists containing the x and y coordinates of the building footprint
    """
    x_coord = lst[0::3]
    y_coord = lst[1::3]
    footprint = []

    for x, y in zip( x_coord, y_coord ):
        pair = [float( x ), float( y )]
        footprint += [pair]

    return footprint


def removeDuplicatesList( lst ):
    """
    Remove the duplicated items of a list
    """
    # 1. Convert into list of tuples
    tpls = [tuple( x ) for x in lst]
    # 2. Create dictionary with empty values and
    # 3. convert back to a list (dups removed)
    dct = list( dict.fromkeys( tpls ) )
    # 4. Convert list of tuples to list of lists
    dup_free = [list( x ) for x in dct]

    return dup_free


def readSector( path ):
    """
    This function read a gml file and create a dictionary with the information of buildings. The
    keys are the building id and the value is another dictionary with some parameters of the building.

    Args:
            path (string): path of the gml file

    Returns:
            my_dict (dictionary): dictionary with the information of buildings
    """
    tree = ET.parse( path )
    root = tree.getroot()

    my_dict = {}

    for city_object in root.findall( '{http://www.opengis.net/citygml/2.0}cityObjectMember' ):

        building = city_object.find( "{http://www.opengis.net/citygml/building/2.0}Building" )
        building_id = building.attrib[ '{http://www.opengis.net/gml}id' ]
        coord_building = []
        deviation_min = 999999

        if city_object.find( "{http://www.opengis.net/citygml/building/2.0}Building/"
                            "{http://www.opengis.net/citygml/building/2.0}consistsOfBuildingPart" ):

            for elem in city_object.findall( "{http://www.opengis.net/citygml/building/2.0}Building/"
                                             "{http://www.opengis.net/citygml/building/2.0}consistsOfBuildingPart/"
                                             "{http://www.opengis.net/citygml/building/2.0}BuildingPart/"
                                             "{http://www.opengis.net/citygml/building/2.0}lod1Solid/"
                                             "{http://www.opengis.net/gml}Solid/"
                                             "{http://www.opengis.net/gml}exterior/"
                                             "{http://www.opengis.net/gml}CompositeSurface" ):

                for coord in elem.findall( "{http://www.opengis.net/gml}surfaceMember/"
                                           "{http://www.opengis.net/gml}Polygon/"
                                           "{http://www.opengis.net/gml}exterior/"
                                           "{http://www.opengis.net/gml}LinearRing/"
                                           "{http://www.opengis.net/gml}posList" ):

                    polygon_face = [float( x ) for x in coord.text.split( sep=' ' )]
                    face_deviation = statistics.stdev( polygon_face[2::3] )

                    if face_deviation < deviation_min:
                        footprint = polygon_face
                        deviation_min = face_deviation

#                     coord_building += coord.text.split( sep=' ' )
            coord_building += footprint

        else:
            for elem in city_object.findall( "{http://www.opengis.net/citygml/building/2.0}Building/"
                                             "{http://www.opengis.net/citygml/building/2.0}lod1Solid/"
                                             "{http://www.opengis.net/gml}Solid/"
                                             "{http://www.opengis.net/gml}exterior/"
                                             "{http://www.opengis.net/gml}CompositeSurface" ):

                for coord in elem.findall( "{http://www.opengis.net/gml}surfaceMember/"
                                           "{http://www.opengis.net/gml}Polygon/"
                                           "{http://www.opengis.net/gml}exterior/"
                                           "{http://www.opengis.net/gml}LinearRing/"
                                           "{http://www.opengis.net/gml}posList" ):

                    polygon_face = [float( x ) for x in coord.text.split( sep=' ' )]
                    face_deviation = statistics.stdev( polygon_face[2::3] )

                    if face_deviation < deviation_min:
                        footprint = polygon_face
                        deviation_min = face_deviation

#                     coord_building += coord.text.split( sep=' ' )
            coord_building += footprint

        footprint_duplicates = footprintListOfLists( coord_building )
#         building_footprint = removeDuplicatesList( footprint_duplicates )
        building_footprint = footprint_duplicates
        building_height = float( building.find( "{http://www.opengis.net/citygml/building/2.0}"
                                                "measuredHeight" ).text )

        my_dict[building_id] = {}
        my_dict[building_id]['footprint'] = building_footprint
        my_dict[building_id]['height'] = building_height

    return my_dict


def centroidnp( footprint ):
    """
    Compute the centroid of the building footprint
    Args:
            footprint (list)
    Return:
            x coordinate (float)
            y coordinate (float)
    """
    x_polygon = []
    y_polygon = []
    for point in footprint:
        x_polygon += [point[0]]
        y_polygon += [point[1]]
    length = len( x_polygon )
    sum_x = np.sum( x_polygon )
    sum_y = np.sum( y_polygon )
    return sum_x / length, sum_y / length


def addCentroid2Dict( building_dict ):
    """
    Add the centroid to the dictionary with the building information

    Args:
            building_dict (dictionary): dictionary containing the information of all the buildings
    Returns:
            building_dict (dictionary): dictionary containing the information of all the buildings
    """
    building_df = pd.DataFrame.from_dict( building_dict, orient='index' )
    building_df['centroid'] = building_df['footprint'].apply( lambda footprint:
                                                              centroidnp( footprint ) )

    transformer = Transformer.from_crs( "EPSG:25832", "EPSG:4326", always_xy=True )

    building_df['centroid_latlon'] = building_df['centroid'].apply( lambda centroid:
                                                             transformer.transform( centroid[0],
                                                                                    centroid[1] ) )
    building_dict = building_df.to_dict( orient='index' )

    return building_dict


def readCity( directory ):
    """
    Create a dictionary with the information of all the buildings included in the gml files

    Args:
            directory (string): directory where the gml files are stored
    Returns:
            building_dict (dictionary): dictionary containing the information of all the buildings
    """

    gml_files = gmlFiles( directory )

    building_dict = {}
    print( 'Reading the building data...' )
    for path in gml_files:

        building_dict.update( readSector( path ) )

    print( 'Calculating centroids...' )
    building_dict = addCentroid2Dict( building_dict )

    return building_dict


if __name__ == '__main__':
#     path = r"C:\Users\jbueno\Desktop\Stadtmodell_Hannover_CityGML_LoD1\CityGML_LoD1\5410_5806.gml"
#     print( readSector( path ) )
#     directory = r"C:\Users\jbueno\Desktop\Stadtmodell_Hannover_CityGML_LoD1\CityGML_LoD1"
    directory = r"C:\workspace3\USEPE_test\src\resources\cityModels\Stadtmodell_Hannover_CityGML_LoD1"
#    directory = r"C:\Users\jbueno\Desktop\Stadtmodell_Hannover_CityGML_LoD1\Tests"
    building_dict = readCity( directory )
    print( len( building_dict ) )
    print( building_dict )

