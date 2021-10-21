#!/usr/bin/python

"""
A module to define the sectors into which the city is divided to consider the building height
"""

__author__ = 'jbueno'
__copyright__ = '(c) Nommon 2021'


import pandas as pd


def defSectors( lon_min, lon_max, lat_min, lat_max, divisions ):
    """
    Define sectora
    Args:
            lon_min (float): float representing the minimum longitude of the region where building altitude is considered.
            lon_max (float): float representing the maximum longitude of the region where building altitude is considered.
            lat_min (float): float representing the minimum latitude of the region where building altitude is considered.
            lat_max (float): float representing the maximum latitude of the region where building altitude is considered.
            divisions (integer): integer representing the number of horizontal and vertical airspace divisions.
    Returns:
            sectors (DataFrame): dataframe with the sector information
    """
    delta_lon = ( lon_max - lon_min ) / divisions
    delta_lat = ( lat_max - lat_min ) / divisions

    rows_list = []
    for i in range( divisions ):
        for j in range( divisions ):
            my_dict = ( {'sector': 'sector' + str( i * divisions + j ),
                         'lon_min': lon_min + j * delta_lon,
                         'lon_max': lon_min + ( j + 1 ) * delta_lon,
                         'lat_min': lat_min + i * delta_lat,
                         'lat_max': lat_min + ( i + 1 ) * delta_lat} )
            rows_list.append( my_dict )

    sectors = pd.DataFrame( rows_list )

    return sectors


def sectorContainPoint( sectors, point ):
    """
    Determine the sector to which the point belongs

    Args:
            sectors (DataFrame): dataframe with the sector information
            point (list): point coordinates

    Returns:
            sector (string): sector to which the point belongs
    """
    sector_row = sectors.loc[( sectors['lon_min'] <= point[0] ) &
                             ( sectors['lon_max'] > point[0] ) &
                             ( sectors['lat_min'] <= point[1] ) &
                             ( sectors['lat_max'] > point[1] )]

    if sector_row.empty:
        sector = 'External'
    else:
        sector = sector_row['sector'].values[0]

    return sector

def addSector2Building( building_dict, sectors ):
    """
    Add the sector to which the building belongs

    Args:
            building_dict (dictionary): building information
            sectors (DataFrame): dataframe with the sector information

    Returns:
            building_dict (dictionary): building information with the new parameter "sector"
    """
    building_df = pd.DataFrame.from_dict( building_dict, orient='index' )

    building_df['sector'] = building_df['centroid_latlon'].apply( lambda centroid:
                                                                  sectorContainPoint( sectors,
                                                                                      centroid ) )

    building_dict = building_df.to_dict( orient='index' )

    return building_dict


def defSectorsAltitude( sectors, building_dict ):
    """
    Determine the altitude limit of each sector

    Args:
            sectors (DataFrame): dataframe with the sector information
            building_dict (dictionary): building information

    Returns:
            sectors (DataFrame): dataframe with the sector information with the new column "altitude_limit"
    """
    df = pd.DataFrame.from_dict( building_dict, orient='index' )
    sector_altitude_list = []
    for sector in sectors['sector'].values:
        df_aux = df[df['sector'] == sector]
        if df_aux.empty:
            sector_altitude = 0
        else:
            sector_altitude = max( df_aux['height'].values )

        sector_altitude_list.append( sector_altitude )

    sectors['altitude_limit'] = sector_altitude_list

    return sectors


def mainSectorsLimit( lon_min, lon_max, lat_min, lat_max, divisions, building_dict ):
    """
    Create the division in sectors, assign a sector to each building and define the altitude of
    sectors.

    Args:
            lon_min (float): float representing the minimum longitude of the region where building altitude is considered.
            lon_max (float): float representing the maximum longitude of the region where building altitude is considered.
            lat_min (float): float representing the minimum latitude of the region where building altitude is considered.
            lat_max (float): float representing the maximum latitude of the region where building altitude is considered.
            divisions (integer): integer representing the number of horizontal and vertical airspace divisions.
            building_dict (dict): dictionary with information about buildings

    Returns:
            sectors (DataFrame): dataframe with the sector information
            building_dict: dictionary with information about buildings
    """
    print( 'Assigning altitude to sectors...' )
    sectors = defSectors( lon_min, lon_max, lat_min, lat_max, divisions )

    building_dict = addSector2Building( building_dict, sectors )

    sectors = defSectorsAltitude( sectors, building_dict )

    return sectors, building_dict


if __name__ == '__main__':
    from building_height import readCity
    directory = r"C:\Users\jbueno\Desktop\Stadtmodell_Hannover_CityGML_LoD1\Tests"
    building_dict = readCity( directory )
    print( building_dict )
    result = mainSectorsLimit( 9, 10, 40, 60, 4, building_dict )
    sectors = result[0]
    building_dict = result [1]
    print( building_dict )
    print( sectors )
