""" 3D Terrain and Skyline Plugin for BlueSky. 
Version 1.0
ToDo: GPWS
ToDo: say 50, 40, 30, 20, 10, retard!
Created by Yaofu Zhou"""
import os
import numpy as np
import datetime
import pandas as pd
from numpy.random import SeedSequence, default_rng
import matplotlib.pyplot as plt
import time

# Import the global bluesky objects. Uncomment the ones you need
from bluesky import core, stack, traf, settings, resource  #, navdb, sim, scr, tools
from bluesky.tools.misc import txt2lat, txt2lon


DEM_resolution = settings.DEM_resolution
# for ~30 meter resolution or "SRTMGL3" for ~90 meter resolution.
# These are the two options.

DEM_data_path = str(resource(settings.navdata_path))

meter2feet = 3.280839895

# opentopography_api_key = settings.opentopography_api_key

from terrain_and_skyline_pkg.init_terrain import terrain
stack.stack("ECHO Interpolation function for terrain elevation ready.")
stack.stack(f"ECHO Terrain elevation available for: "
            + f"S={settings.DEM_south} W={settings.DEM_west} "
            + f"N={settings.DEM_north} E={settings.DEM_east}")


### Initialization function of your plugin. Do not change the name of this
### function, as it is the way BlueSky recognises this file as a plugin.
def init_plugin():
    ''' Plugin initialisation function. '''
    # Instantiate our example entity
    elevation = Elevation()

    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'TERRAIN',

        # The type of this plugin.
        'plugin_type':     'sim',
        }

    # init_plugin() should always return a configuration dict.
    return config


### Entities in BlueSky are objects that are created only once
### (called singleton)
### which implement some traffic or other simulation functionality.
### To define an entity that ADDS functionality to BlueSky, create a class that
### inherits from bluesky.core.Entity.
### To replace existing functionality in BlueSky, inherit from the class that
### provides the original implementation (see for example the asas/eby plugin).
class Elevation(core.Entity):
    ''' Example new entity object for BlueSky. '''

    def __init__(self):
        super().__init__()
        # All classes deriving from Entity can register lists and numpy arrays
        # that hold per-aircraft data. This way, their size is automatically
        # updated when aircraft are created or deleted in the simulation.
        with self.settrafarrays():
            # abv_gnd_lvl means above-ground level, or AGL
            self.abv_gnd_lvl = np.array([])


    def create(self, n=1):
        ''' This function gets called automatically when a aircraft is
        created. '''
        # Don't forget to call the base class create when you reimplement this
        # function!
        super().create(n)
        # For the new aircraft, the above ground level is the altitude (above
        # mean sea level) minus the terrain elevation.
        self.abv_gnd_lvl[-n:] = traf.alt[-n:] \
            - terrain.point_terrain(traf.lat[-n:], traf.lon[-n:])
        stack.stack(f"ECHO AGL of {traf.id[-n]} is {self.abv_gnd_lvl[-n]:.0f} "
                    + f"meters or {(self.abv_gnd_lvl[-n] * meter2feet):.0f} "
                    + "feet.")

    # Functions that need to be called periodically can be indicated to BlueSky
    # with the timed_function decorator
    @core.timed_function(name='elevation', dt=settings.performance_dt)
    def update(self):
        ''' Periodic update function for the elevation entity. '''
        for i in range(len(self.abv_gnd_lvl)):
            self.abv_gnd_lvl[i] = traf.alt[i] \
                - terrain.point_terrain(traf.lat[i], traf.lon[i])

        traf.agl = self.abv_gnd_lvl


    @stack.command(name="AGL", aliases=("RADARALTIMETER", "RA", "RALT",
                                        "LRRA"))
    def agl(self, acid):
        # Aircraftc callsigns in bs are capitalized.
        acidx = traf.id.index(acid.upper())
        msl = traf.alt[acidx]
        traf.agl[acidx] \
            = msl - terrain.point_terrain(traf.lat[acidx], traf.lon[acidx])
        agl = traf.agl[acidx]
        agl_in_feet = agl * meter2feet

        if not settings.agl_mute:
            os.system(f"say {agl_in_feet:.0f}.")

        return True, f"AGL of {acid.upper()} is {agl:.0f} meters or " \
            + f"{agl_in_feet:.0f} feet."


    # You can create new stack commands with the stack.command decorator.
    # By default, the stack command name is set to the function name.
    # The default argument type is a case-sensitive word. You can indicate different
    # types using argument annotations.
    @stack.command(name="GROUNDMSL")
    def groundmsl(self, *args):
        ''' Print the Mean Sea Level of the ground at lat, lon
        or Aircraft callsign.

        ToDo: Accept waypoint, airport, runway, etc. as input.'''

        if len(args) == 2:
            lat = txt2lat(args[0])
            lon = txt2lon(args[1])
            gnd_ele_in_meter = terrain.point_terrain(lat, lon)
            gnd_ele_in_feet = gnd_ele_in_meter * meter2feet
            return True, f"The ground elevation at {lat, lon} is " \
                + f"{gnd_ele_in_meter:.0f} meters or " \
                + f"{gnd_ele_in_feet:.0f} feet."

        elif len(args) == 1:
            # Aircraftc callsigns in bs are capitalized.
            acid = traf.id.index(args[0].upper())
            gnd_ele_in_meter = terrain.point_terrain(traf.lat[acid],
                                                     traf.lon[acid])
            gnd_ele_in_feet = gnd_ele_in_meter * meter2feet
            return True, f"The ground elevation below {args[0].upper()} is " \
                + f"{gnd_ele_in_meter:.0f} meters or " \
                + f"{gnd_ele_in_feet:.0f} feet."


    @stack.command(name="SHOWELEMAP")
    def print_terrain_map(self):
        """A testing feature that print the elevation map in .pdf and .png
        formats."""

        num_points = settings.num_points_drawn
        stack.stack("ECHO Stand by for elevation map generated using "
                    + f"{num_points} random (lat, lon) points...\n"
                    + "See Termianl for additional information.")
        y_min = settings.DEM_south
        x_min = settings.DEM_west
        y_max = settings.DEM_north
        x_max = settings.DEM_east
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

        # To remove edge anomaly in the drawing.
        y_min -= 0.001
        x_min -= 0.001
        y_max += 0.001
        x_max += 0.001

        print(f"Creating test elevation map using {num_points} random (lat, lon) "
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
              + f"{out_filename}.png \nand\n"
              + f"{out_filename}.pdf...")
        fig.savefig(out_filename+'.png', dpi=600, format='png')
        fig.savefig(out_filename+'.pdf', format='pdf')

        ax.w_xaxis.set_pane_color((0., 0., 0., 0.))
        ax.w_yaxis.set_pane_color((0., 0., 0., 0.))
        ax.w_zaxis.set_pane_color((0., 0., 0., 0.))

        # print("Showing...")
        # os.system("say 'done done done done done done done done done done'")
        # plt.show(block=False)
        return True, f"ECHO Elevation map saved to\n{out_filename}.png\n" \
            + f"and\n{out_filename}.pdf..."


if __name__ == "__main__":
    pass
#     south = settings.DEM_south
#     west = settings.DEM_west
#     north = settings.DEM_north
#     east = settings.DEM_east
# 
#     bounding_box = (south, west, north, east)
# 
#     elevation.print_elevation_map()
