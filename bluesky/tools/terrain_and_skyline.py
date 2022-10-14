import os
import datetime
import numpy as np
import pandas as pd
from numpy.random import SeedSequence, default_rng
import matplotlib.pyplot as plt
import time

from init_terrain import terrain, DEM_resolution, DEM_data_path
# from init_skyline import skyline

# DEM_resolution = "SRTMGL1"
reset_terrain = False
# DEM_data_path = "DEM_data/"


def print_elevation_map(bounding_box):
    """A testing feature that """

    y_min, x_min, y_max, x_max = bounding_box
    W, E = x_min, x_max
    S, N = y_min, y_max
    drawing_name = "Randomly Generated Elevation Map with Dataset=" \
        + f"{DEM_resolution},\n" \
        + f"W={x_min}, E={x_max}, S={y_min}, N={y_max} [deg]."
    out_filename = f"{DEM_data_path}/{DEM_resolution}_{W}_{E}_{S}_{N}"
    # This section is for placing the rectangular bounding box in the middle of
    # a square.
    delta_x = x_max - x_min
    delta_y = y_max - y_min
    delta_lon2delta_lat = delta_x/delta_y
    # Same delta longitude does not correspond to same distance at different
    # latitudes. The shrinkage factor compared to that at the Equator is:
    cos_lat = np.cos((y_max + y_min)*np.pi/180./2)
    # This scaling is needed because the 3D plot in use is always presented
    # in a cube
    if cos_lat*delta_lon2delta_lat > 1.:
        y_max += delta_y/2. * (cos_lat*delta_lon2delta_lat - 1.)
        y_min -= delta_y/2. * (cos_lat*delta_lon2delta_lat - 1.)
    else:
        x_max += delta_x/2. * (1./cos_lat/delta_lon2delta_lat - 1.)
        x_min -= delta_x/2. * (1./cos_lat/delta_lon2delta_lat - 1.)

    drawing_df = pd.DataFrame(columns=['x', 'y', 'z'])

    num_points = 100000
    print(f"Creating test elevation map using {num_points} random (lat, lon)"
          + "points")
    # Initiate a random number generate.
    _ss = SeedSequence(entropy=int(
        datetime.datetime.now().strftime("%Y%m%d%H%M%S")))
    seed_list = _ss.spawn(1)  # The list of RNGs has only one RNG in this case.
    rng_list = [default_rng(s) for s in seed_list]

    time_start = time.perf_counter()
    i = 0
    while i < num_points:
        # Random points within area of interest
        x_rand = x_min + rng_list[0].random()*(x_max - x_min)
        y_rand = y_min + rng_list[0].random()*(y_max - y_min)
        # Points outside area of interest are given elevation of 0
        if x_rand > E or x_rand < W or y_rand > N or y_rand < S:
            z_rand = 0.
        else:
            z_rand = terrain.point_terrain(y_rand, x_rand)

        point_df = pd.DataFrame([[x_rand, y_rand, z_rand]],
                                columns=['x', 'y', 'z'])
        drawing_df = pd.concat([drawing_df, point_df], ignore_index=True)
        i += 1
        if i % (num_points/10) == 0:
            print(i)
    time_finish = time.perf_counter()
    time_spent = time_finish - time_start
    print(f"Time to plot {num_points} is {time_spent} sec.")
    print(f"Each elevation call takes less than {time_spent/num_points} sec.")

    drawing_df.sort_values(["x", "y", "z"], inplace=True, ignore_index=True)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_trisurf(drawing_df.x, drawing_df.y, drawing_df.z,
                           shade=True, linewidth=0.1)
    ax.set_xlabel('Longitude [deg]')
    ax.set_ylabel('Latitude [deg]')
    ax.set_zlabel('Elevation [m]')
    plt.title(drawing_name)

    print("Saving test elevation map to\n"
          + f"{out_filename}.png \n and\n"
          + f"{out_filename}.pdf...")
    fig.savefig(out_filename+'.png', dpi=600, format='png')
    fig.savefig(out_filename+'.pdf', format='pdf')

    print("Showing...")
    # ax.w_xaxis.set_pane_color((0.1, 0.1, 0.1, 0.5))
    # ax.w_yaxis.set_pane_color((0.1, 0.1, 0.1, 0.5))
    # ax.w_zaxis.set_pane_color((0.1, 0.1, 0.1, 0.5))
    ax.w_xaxis.set_pane_color((0., 0., 0., 0.))
    ax.w_yaxis.set_pane_color((0., 0., 0., 0.))
    ax.w_zaxis.set_pane_color((0., 0., 0., 0.))

    os.system("echo 'done'")
    os.system("say 'done done done done done done done done done done done'")

    plt.show()


if __name__ == "__main__":
    # Greater NYC for debug and test
    south = 40.2
    west = -74.5
    north = 41.5
    east = -73.1
    bounding_box = (south, west, north, east)

    print_elevation_map(bounding_box)
