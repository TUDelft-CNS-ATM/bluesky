#!/usr/bin/python

"""
A module to preprocess the wind data
"""
import math
import os
from scipy.interpolate import griddata
import netCDF4 as nc
import numpy as np
import time as tim


__author__ = 'jbueno'
__copyright__ = '(c) Nommon 2021'


def readData( path ):
    """
    Read the wind data.

    Args:
            path (string): indicating the path where the file is located

    Returns:
            wind(netCDF4): wind information. Variables:
                            lat(y,x)
                            latu(y,xu)
                            latv(yv,x)
                            lon(y,x)
                            lonu(y,xu)
                            lonv(yv,x)
                            u(time,zu_3d,y,xu)
                            v(time,zu_3d,yv,x)
                            w(time,zw_3d,y,x)
    """

    wind = nc.Dataset( path )
    return wind


def removedMaskedValues( wind, time ):
    """
    Remove the masked value from the wind data.
    Transform the wind data to the appropriate format for interpolation.

    Args:
            wind(netCDF4): wind information. Variables:
                            lat(y,x)
                            latu(y,xu)
                            latv(yv,x)
                            lon(y,x)
                            lonu(y,xu)
                            lonv(yv,x)
                            u(time,zu_3d,y,xu)
                            v(time,zu_3d,yv,x)
                            w(time,zw_3d,y,x)
            time (integer): index indicating the time of all the time instant included in the data

    Returns:
            points_u (array): coordinates of the grid for the values of the u component
            values_u (array): values of the u component associated to the points_u
            points_v (array): coordinates of the grid for the values of the v component
            values_v (array): values of the v component associated to the points_v
            points_w (array): coordinates of the grid for the values of the w component
            values_w (array): values of the w component associated to the points_w

    """

    x = list( wind['x'][:].data )
    xu = list( wind['xu'][:].data )
    y = list( wind['y'][:].data )
    yv = list( wind['yv'][:].data )
    zu = list( wind['zu_3d'][:].data )
    zw = list( wind['zw_3d'][:].data )

    # u(time,zu_3d,y,xu) --> u(time,zu_3d,y,x)
    print( 'Removing masked values...' )

    combs_list = np.argwhere( wind['u'][time,:].data != wind['u'][:].fill_value )

    points_u = []
    values_u = []
    u = wind.variables['u'][:]
    for comb in combs_list:
        zu_iter = comb[0]
        y_iter = comb[1]
        x_iter = comb[2]
        points_u.append( [zu[zu_iter], y[y_iter], xu[x_iter]] )
        values_u.append( u[time, zu_iter, y_iter, x_iter] )

    points_u = np.array( points_u )
    values_u = np.array( values_u )

    combs_list = np.argwhere( wind['v'][time,:].data != wind['v'][:].fill_value )

    points_v = []
    values_v = []
    v = wind.variables['v'][:]
    for comb in combs_list:
        zu_iter = comb[0]
        y_iter = comb[1]
        x_iter = comb[2]
        points_v.append( [zu[zu_iter], yv[y_iter], x[x_iter]] )
        values_v.append( v[time, zu_iter, y_iter, x_iter] )

    points_v = np.array( points_v )
    values_v = np.array( values_v )

    combs_list = np.argwhere( wind['w'][time,:].data != wind['w'][:].fill_value )

    points_w = []
    values_w = []
    w = wind.variables['w'][:]
    for comb in combs_list:
        zw_iter = comb[0]
        y_iter = comb[1]
        x_iter = comb[2]
        points_w.append( [zw[zw_iter], y[y_iter], x[x_iter]] )
        values_w.append( w[time, zw_iter, y_iter, x_iter] )

    points_w = np.array( points_w )
    values_w = np.array( values_w )

    return points_u, values_u, points_v, values_v, points_w, values_w


def interpolateWind( points_u, values_u, points_v, values_v, points_w, values_w, grid_z, grid_y,
                     grid_x ):
    """
    Interpolate the wind data of the instant defined by "time". The wind is interpolated in the
    points defined by grid_z, grid_y, grid_x.

    Args:
            points_u (array): coordinates of the grid for the values of the u component
            values_u (array): values of the u component associated to the points_u
            points_v (array): coordinates of the grid for the values of the v component
            values_v (array): values of the v component associated to the points_v
            points_w (array): coordinates of the grid for the values of the w component
            values_w (array): values of the w component associated to the points_w
            grid_z (array): z values of the grid where the wind is interpolated
            grid_y (array): y values of the grid where the wind is interpolated
            grid_x (array): x values of the grid where the wind is interpolated
    Returns:
            grid_z (array): z values of the grid where the wind is interpolated
            grid_y (array): y values of the grid where the wind is interpolated
            grid_x (array): x values of the grid where the wind is interpolated
            grid_u (array): wind velocity in the x direction
            grid_v (array): wind velocity in the y direction
            grid_w (array): wind velocity in the z direction
    """

    print( "interpolating..." )
    grid_u = griddata( points_u, values_u, ( grid_z, grid_y, grid_x ), method='nearest' )
    grid_v = griddata( points_v, values_v, ( grid_z, grid_y, grid_x ), method='nearest' )
    grid_w = griddata( points_w, values_w, ( grid_z, grid_y, grid_x ), method='nearest' )

    return grid_z, grid_y, grid_x, grid_u, grid_v, grid_w


def interpolateLat( wind, grid_y, grid_x ):
    """
    Interpolate latitude coordinates in the points defined by grid_y and grid_x.

    Args:
            wind (netCDF4): wind information.
            grid_y (array): y values of the grid where the wind is interpolated
            grid_x (array): x values of the grid where the wind is interpolated
    Returns:
            grid_lat (array): latitude values of the grid where the wind is interpolated
    """
    x = list( wind['x'][:].data )
    y = list( wind['y'][:].data )

    lat = wind.variables['lat'][:]

    combs_list = np.argwhere( lat != lat.fill_value )

    points_lat = []
    values_lat = []
    for comb in combs_list:
        y_iter = comb[0]
        x_iter = comb[1]
        points_lat.append( [y[y_iter], x[x_iter]] )
        values_lat.append( lat[y_iter, x_iter] )

    points_lat = np.array( points_lat )
    values_lat = np.array( values_lat )

    grid_lat = griddata( points_lat, values_lat, ( grid_y, grid_x ), method='nearest' )

    return grid_lat


def interpolateLon( wind, grid_y, grid_x ):
    """
    Interpolate longitude coordinates in the points defined by grid_y and grid_x.

    Args:
            wind (netCDF4): wind information.
            grid_y (array): y values of the grid where the wind is interpolated
            grid_x (array): x values of the grid where the wind is interpolated
    Returns:
            grid_lon (array): longitude values of the grid where the wind is interpolated
    """
    x = list( wind['x'][:].data )
    y = list( wind['y'][:].data )

    lon = wind.variables['lon'][:]

    combs_list = np.argwhere( lon != lon.fill_value )

    points_lon = []
    values_lon = []
    for comb in combs_list:
        y_iter = comb[0]
        x_iter = comb[1]
        points_lon.append( [y[y_iter], x[x_iter]] )
        values_lon.append( lon[y_iter, x_iter] )

    points_lon = np.array( points_lon )
    values_lon = np.array( values_lon )

    grid_lon = griddata( points_lon, values_lon, ( grid_y, grid_x ), method='nearest' )

    return grid_lon


def createWindScenario( scenario_file, grid_lat, grid_lon, grid_z, grid_u, grid_v, grid_w ):
    """
    Create a BlueSky scenario containing the commands to import the wind.

    Args:
            scenario_file (object): text file object where the commands are written
            grid_lat (array): latitude values of the grid where the wind is interpolated
            grid_lon (array): longitude values of the grid where the wind is interpolated
            grid_z (array): z values of the grid where the wind is interpolated
            grid_u (array): wind velocity in the x direction
            grid_v (array): wind velocity in the y direction
            grid_w (array): wind velocity in the z direction
    """
    time = "00:00:00.00"
    m2ft = 3.28084
    ms2knots = 1.94384
    for y_iter in range( grid_z.shape[1] ):
        for x_iter in range( grid_z.shape[2] ):
            lat = grid_lat[y_iter, x_iter]
            lon = grid_lon[y_iter, x_iter]

            new_line = '{0} > WIND {1}, {2}'.format( time, lat, lon )

            for z_iter in range( grid_z.shape[0] ):
                alt = grid_z[z_iter, y_iter, x_iter] * m2ft
                u = grid_u[z_iter, y_iter, x_iter] * ms2knots
                v = grid_v[z_iter, y_iter, x_iter] * ms2knots
                w = grid_w[z_iter, y_iter, x_iter] * ms2knots

                direction = ( math.degrees( math.atan2( u, v ) ) + 180. )
                wind_vector = np.array( [u, v] )
                spd = np.sqrt( wind_vector.dot( wind_vector ) )

                new_line += ', {0}, {1}, {2}'.format( alt, direction, spd )

            scenario_file.write( new_line + '\n' )


def main( path, grid_spacing_list, time ):
    """
    Main function to generate BlueSky scenarios with the wind commands.
    """
    # ------------ Defined by the user ------------------------------
    # path = r".\data\test_hannover_1m_3d.000.nc"
    # path = r".\data\test_hannover_1m_masked_M03.000.nc"
    # grid_spacing_list = [5, 10, 20, 50]
    # time = 0
    # ----------------------------------------------------------------

    print( "Reading data..." )
    wind = readData( path )

    points_u, values_u, points_v, values_v, points_w, values_w = removedMaskedValues( wind, time )

    for grid_spacing in grid_spacing_list:
        scenario_path = r".\scenario\wind_test_{0}m_{1}s.scn"\
            .format( grid_spacing, time )

        x = list( wind['x'][:].data )
        y = list( wind['y'][:].data )
        zu = list( wind['zu_3d'][:].data )

        print( "Creating grid..." )
        grid_z, grid_y, grid_x = np.mgrid[zu[0]:zu[-1]:grid_spacing,
                                          y[0]:y[-1]:grid_spacing,
                                          x[0]:x[-1]:grid_spacing]

        print( "Interpolating wind velocity..." )
        grid_z, grid_y, grid_x, grid_u, grid_v, grid_w = interpolateWind( 
            points_u, values_u, points_v, values_v, points_w, values_w, grid_z, grid_y, grid_x )

        print( "Interpolating latitude and longitude..." )
        grid_lat = interpolateLat( wind, grid_y[0], grid_x[0] )
        grid_lon = interpolateLon( wind, grid_y[0], grid_x[0] )

        if not os.path.exists( os.path.dirname( scenario_path ) ):
            os.makedirs( os.path.dirname( scenario_path ) )

        scenario_file = open( scenario_path, 'w' )

        print( "Writing scenario..." )
        createWindScenario( scenario_file, grid_lat, grid_lon, grid_z, grid_u, grid_v, grid_w )

        scenario_file.close()
    print( "Finish" )


if __name__ == '__main__':
    main()
