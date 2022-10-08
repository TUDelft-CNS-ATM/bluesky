import os
import glob
import requests
import numpy as np
import pandas as pd
from itertools import product
from scipy import interpolate
import math

from osgeo import gdal

from bluesky import resource
from bluesky import settings

from .decorators import timer

settings.set_variable_defaults(opentopography_api_key="demoapikeyot2022",
                               DEM_resolution="SRTMGL1",
                               DEM_data_path=str(
                                   resource(settings.navdata_path)),
                               reset_terrain=False)

south = settings.DEM_south
west = settings.DEM_west
north = settings.DEM_north
east = settings.DEM_east
bounding_box = (south, west, north, east)

DEM_resolution = settings.DEM_resolution
# for ~30 meter resolution or "SRTMGL3" for ~90 meter resolution.
# These are the two options.

opentopography_api_key = settings.opentopography_api_key
DEM_data_path = str(resource(settings.navdata_path))


class DEM():
    """A singleton class to provide functions that yield terrain elevation
    anywhere in the interested area.

    DEM stands for Digital Surface Model"""

    def __init__(self, bounding_box, reset=False):
        if DEM_resolution not in ("SRTMGL1", "SRTMGL3"):
            raise ValueError("DEM_resolution must be either 'SRTMGL1' or "
                             + "'SRTMGL3'. DEM_resolution entered: "
                             + f"{DEM_resolution}")
        self.bounding_box = bounding_box
        self.y_min, self.x_min, self.y_max, self.x_max = bounding_box
        angular_area = abs(self.x_max - self.x_min)*(self.y_max - self.y_min) \
            * math.cos((self.x_max + self.x_min)*np.pi/180./2.)
        print("\nSelected angular area = ", angular_area, "deg^2.")
        if DEM_resolution == "SRTMGL1" and angular_area > 0.6:
            input("\nSelected area > 7400 km^2, or 2160 nautical mile^2, "
                  + "or 0.6 deg^2. The feature of terrain elevations with"
                  + f"dataset {DEM_resolution} can be very slow.\n"
                  + "Abort or press ENTER to continue...\n")
        if DEM_resolution == "SRTMGL3" and angular_area > 5.3:
            input("\nSelected area > 65300 km^2, or 19000 nautical mile^2, "
                  + "or 5.3 deg^2. The feature of terrain elevations with"
                  + f"dataset {DEM_resolution} can be very slow.\n"
                  + "Abort or press ENTER to continue...\n")
        self.reset = reset
        self.needed_list = []
        self.download_list = []
        if self._needs_download():
            self._download_DEM()
        self.interpo_terrain = self._init_interpo_DEM()

    def _needs_download(self):
        """Checks if DEM_data_path contains files that cover the requested
        bounding_box. The datasets are handled in 1 deg latitude
        X 1 deg longitude sections.

        This method set the self.download_list to
        include all necessary sections to cover the requested bounding_box.

        Returns True if self.reset is set True even if needed files are
        present."""

        download_needed = False

        download_range_W = int(math.floor(self.x_min))
        download_range_E = int(math.ceil(self.x_max))
        download_range_S = int(math.floor(self.y_min))
        download_range_N = int(math.ceil(self.y_max))

        if download_range_W not in range(-180, 180) \
           or download_range_E not in range(-180, 180):
            raise ValueError("Longitude should be in [-180, 180)")
        if download_range_S not in range(-90, 90) \
           or download_range_N not in range(-90, 90):
            raise ValueError("Latitide should be in [-90, 90]")

        # Expecting integer part of longitude in [-180, 180).
        # This if statement, together with the next, handles the reagion across
        # 180/-180 deg longitudes.
        if download_range_W > download_range_E:
            download_range_E = download_range_E + 360
        for x in range(download_range_W, download_range_E):
            for y in range(download_range_S, download_range_N):
                if x >= 180:
                    # This condition catches the reagion across 180/-180 deg
                    # longitudes.
                    self.needed_list.append([x-360, y, DEM_resolution])
                else:
                    self.needed_list.append([x, y, DEM_resolution])
        print("DEM datasets needed:\n", self.needed_list, "\n")

        # This section works out the datasets already available at
        # DEM_data_path
        available_list = []
        for file in glob.glob(DEM_data_path+"/*.tif"):
            resolution_str, W, E, S, N = file.replace(DEM_data_path, "") \
                .replace("/", "").replace(".tif", "").split("_")
            if resolution_str != DEM_resolution:
                continue
            W = int(math.ceil(float(W)))
            E = int(math.floor(float(E)))
            S = int(math.ceil(float(S)))
            N = int(math.floor(float(N)))
            if W != E and S != N:
                if W > E:
                    E += 360
                for x in range(W, E):
                    for y in range(S, N):
                        if x >= 180:
                            available_list.append([x-360, y, resolution_str])
                        else:
                            available_list.append([x, y, resolution_str])

        already_has_list = [item for item in self.needed_list
                            if item in available_list]
        print("DEM datasets already available:\n", already_has_list, "\n")
        self.download_list = [item for item in self.needed_list
                              if item not in available_list]
        print("DEM datasets to download:\n", self.download_list, "\n")

        if self.download_list or self.reset:
            download_needed = True

        return download_needed

    @timer
    def _download_DEM(self):
        """Downloads all 1 deg latitude X 1 deg longitude sections of data
        specified by self.download_list.

        The coordinate reference system (CRS) here is epsg:4326"""

        # opentopography_api_key should have been set outside this class
        API_KEY = opentopography_api_key

        # opentopography service does not always respect the entire bounding
        # box, so request for a little more. 0.002 corresponds to about 222
        # meters on the Equator or 87 meters at 67 deg N/S latitudes.
        buffer = 0.002

        isExist = os.path.exists(DEM_data_path)
        if not isExist:
            print(f"Creating DEM data directory {DEM_data_path} ...")
            os.makedirs(DEM_data_path, exist_ok=False)
        else:
            print(f"Using existing DEM data directory {DEM_data_path} .")

        for item in self.download_list:
            x, y, resolution_str = item
            # 1 deg latitude X 1 deg longitude section
            W, E = x, x+1
            S, N = y, y+1
            # Filename reflects the set resulution and current section
            out_filename = f"{DEM_data_path}/" \
                + f"{resolution_str}_{W}_{E}_{S}_{N}.tif"

            # To request a little more for each 1 deg latitude X 1 deg
            # longitude section
            W, E = W - buffer, E + buffer
            S, N = S - buffer, N + buffer

            URL = "https://portal.opentopography.org/API/globaldem?demtype=" \
                + f"{resolution_str}&west={W}&east={E}&south={S}&north={N}" \
                + f"&outputFormat=GTiff&API_Key={API_KEY}"

            #  TODO: allow the downloading to be multithreading
            print(f"Downloading dataset {item}...")
            response = requests.get(URL)
            #  TODO: add handling of exceptions

            open(out_filename, "wb").write(response.content)

    @timer
    def _init_interpo_DEM(self):
        """Returns a function named interpo_terrain of the type
        scipy.interpolate.LinearNDInterpolator, which yields a list of
        elevation values corresponding to a list of latitudes and a list of
        longitudes with the same length.

        The coordinate reference system (CRS) here is epsg:4326
        Elevations are in meters."""

        gdal.UseExceptions()

        DEM_info = ['y', 'x', 'z']
        DEM_df = pd.DataFrame(columns=DEM_info)
        for item in self.needed_list:
            x, y, resolution_str = item
            # 1 deg latitude X 1 deg longitude section
            W, E = x, x+1
            S, N = y, y+1
            # Filename reflects the set resulution and current section
            datafile = f"{DEM_data_path}/{resolution_str}_{W}_{E}_{S}_{N}.tif"
            print(f"Reading dateset {item}...")
            ds = gdal.Open(datafile)
            band = ds.GetRasterBand(1)
            # elevation[y, x] = elevation[lat, lon]
            elevation = band.ReadAsArray()

            # longitudes (x) and latitudes (y)
            width = ds.RasterXSize
            height = ds.RasterYSize
            gt = ds.GetGeoTransform()
            minx = gt[0]
            miny = gt[3] + width*gt[4] + height*gt[5]
            maxx = gt[0] + width*gt[1] + height*gt[2]
            maxy = gt[3]

            x = np.linspace(minx, maxx, elevation.shape[1])  # x = longitude
            y = np.linspace(maxy, miny, elevation.shape[0])  # y = latitude
            yx = list(product(y, x))  # ((y0, x0), (y0, x1), (y0, x2), ...)
            z = list(np.concatenate(elevation).flat)
            current_df = pd.DataFrame(yx, columns=["y", "x"])
            current_df["z"] = z
            current_df.drop(current_df[current_df['z'] < -420].index,
                            inplace=True)
            current_df.drop(current_df[current_df['y'] < self.y_min].index,
                            inplace=True)
            current_df.drop(current_df[current_df['x'] < self.x_min].index,
                            inplace=True)
            current_df.drop(current_df[current_df['y'] > self.y_max].index,
                            inplace=True)
            current_df.drop(current_df[current_df['x'] > self.x_max].index,
                            inplace=True)
            DEM_df = pd.concat([DEM_df, current_df], ignore_index=True)

        print("Combining terrain datasets...")
        DEM_df.drop_duplicates(inplace=True, ignore_index=True)
        DEM_df.sort_values(["y", "x"], inplace=True, ignore_index=True)

        x = DEM_df["x"].values.tolist()
        y = DEM_df["y"].values.tolist()
        z = DEM_df["z"].values.tolist()
        print("Initializing interpolation function for terrain elevation...")
        interpo_terrain = interpolate.LinearNDInterpolator(list(zip(y, x)), z,
                                                           rescale=False,
                                                           fill_value=0.)
        # interpo_terrain = interpolate.NearestNDInterpolator(list(zip(y, x)), z,
        #                                                     rescale=False)

        return interpo_terrain

    def point_terrain(self, lat, lon):
        """Returns terrain elevation in meter for given latitude and longitude
        pair.

        The coordinate reference system (CRS) here is epsg:4326
        Elevations are in meters."""

        return self.interpo_terrain([lat], [lon])[0]


# Instantiate a singleton
terrain = DEM(bounding_box)


if __name__ == "__main__":
    # NYC for test and debug
    south = 40.2
    west = -74.5
    north = 41.5
    east = -73.1
    bounding_box = (south, west, north, east)

    print(terrain.point_terrain(south, east))
    os.system("say 'done done done done done done done done done done done'")
