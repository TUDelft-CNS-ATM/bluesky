# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 09:19:29 2021

@author: nino_
"""

import os
import sys
import numpy as np
import pygrib
import requests
import bluesky as bs
from bluesky.tools.aero import ft, kts
from bluesky import settings, stack
from bluesky.core import timed_function
from bluesky.traffic.windsim import WindSim

settings.set_variable_defaults(
    windgfs_url="https://www.ncei.noaa.gov/data/global-forecast-system/access/historical/analysis/")

# nlayer = 23

datadir = settings.data_path + '/grib/'

if not os.path.exists(datadir):
    os.makedirs(datadir)


def init_plugin():
    global windgfs
    windgfs = WindGFS()

    config = {
        'plugin_name': 'WINDGFS',
        'plugin_type': 'sim'
    }

    return config

class WindGFS(WindSim):
    def __init__(self):
        super().__init__()
        print('WINDGFS CTOR')
        self.year = 0
        self.month = 0
        self.day = 0
        self.hour = 0
        self.lat0 = -90
        self.lon0 = -180
        self.lat1 = 90
        self.lon1 = 180

        # Switch for periodic loading of new GFS data
        self.autoload = False


    def fetch_grb(self, year, month, day, hour, pred=0):
        ym = "%04d%02d" % (year, month)
        ymd = "%04d%02d%02d" % (year, month, day)
        hm = "%02d00" % hour
        pred = "%03d" % pred

        remote_loc = "/%s/%s/gfsanl_3_%s_%s_%s.grb2" % (ym, ymd, ymd, hm, pred)

        fname = "gfsanl_3_%s_%s_%s.grb2" % (ymd, hm, pred)
        fpath = datadir + fname

        remote_url = settings.windgfs_url + remote_loc

        if not os.path.isfile(fpath):
            bs.scr.echo("Downloading file, please wait...")
            print("Downloading %s" % remote_url)

            response = requests.get(remote_url, stream=True)

            if response.status_code != 200:
                print("Error. remote data not found")
                return None

            with open(fpath, "wb") as f:
                total_length = response.headers.get('content-length')

                if total_length is None:  # no content length header
                    f.write(response.content)
                else:
                    dl = 0
                    total_length = int(total_length)
                    for data in response.iter_content(chunk_size=4096):
                        dl += len(data)
                        f.write(data)
                        done = int(50 * dl / total_length)
                        sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50-done)) )
                        sys.stdout.flush()

        bs.scr.echo("Download completed.")
        grb = pygrib.open(fpath)

        return grb


    def extract_wind(self, grb, lat0, lon0, lat1, lon1):

        grb_wind_v = grb.select(shortName="v", typeOfLevel=['isobaricInhPa'])
        grb_wind_u = grb.select(shortName="u", typeOfLevel=['isobaricInhPa'])

        lats = np.array([])
        lons = np.array([])
        alts = np.array([])
        vxs = np.array([])
        vys = np.array([])

        for grbu, grbv in zip(grb_wind_u, grb_wind_v):
            level = grbu.level

            if level < 140:  # lesss than 140 hPa, above about 45 k ft
                continue
            else:
                vxs_ = grbu.values
                vys_ = grbv.values

                p = level * 100
                h = (1 - (p / 101325.0)**0.190264) * 44330.76923    # in meters

                lats_ = grbu.latlons()[0].flatten()
                lons_ = grbu.latlons()[1].flatten()
                alts_ = round(h) * np.ones(len(lats_))

                lats = np.append(lats, lats_)
                lons = np.append(lons, lons_)
                alts = np.append(alts, alts_)
                vxs = np.append(vxs, vxs_)
                vys = np.append(vys, vys_)

        lons = (lons + 180) % 360.0 - 180.0     # convert range from 0~360 to -180~180

        lat0_ = min(lat0, lat1)
        lat1_ = max(lat0, lat1)
        lon0_ = min(lon0, lon1)
        lon1_ = max(lon0, lon1)

        mask = (lats >= lat0_) & (lats <= lat1_) & (lons >= lon0_) & (lons <= lon1_)

        data = np.array([lats[mask], lons[mask], alts[mask], vxs[mask], vys[mask]])

        return data

    @stack.command(name='WINDGFS')
    def loadwind(self, lat0: 'lat', lon0: 'lon', lat1: 'lat', lon1: 'lon',
               year: int=None, month: int=None, day: int=None, hour: int=None):
        ''' WINDGFS: Load a windfield directly from NOAA database.
            Arguments:
            - lat0, lon0, lat1, lon1 [deg]: Bounding box in which to generate wind field
            - year, month, day, hour: Date and time of wind data (optional, will use
              current simulation UTC if not specified).
        '''
        self.lat0, self.lon0, self.lat1, self.lon1 =  lat0, lon0, lat1, lon1
        self.year = year or bs.sim.utc.year
        self.month = month or bs.sim.utc.month
        self.day = day or bs.sim.utc.day
        self.hour = hour or bs.sim.utc.hour

        # round hour to 3 hours, check if it is a +3h prediction
        self.hour  = round(self.hour / 3) * 3
        if self.hour in [3, 9, 15, 21]:
            self.hour = self.hour - 3
            pred = 3
        else:
            pred = 0

        txt = "Loading wind field for %s-%s-%s %s:00..." % (self.year, self.month, self.day, self.hour)
        bs.scr.echo("%s" % txt)

        grb = self.fetch_grb(self.year, self.month, self.day, self.hour, pred)

        if grb is None:
            return False, "Wind data not exist in area [%d, %d], [%d, %d]. " \
                % (self.lat0, self.lat1, self.lon0, self.lon1) \
                + "time: %04d-%02d-%02d %02d:00" \
                % (self.year, self.month, self.day, self.hour)

        # first clear exisiting wind field
        self.clear()

        # add new wind field
        data = self.extract_wind(
            grb, self.lat0, self.lon0, self.lat1, self.lon1).T
        # Sort by lat, lon, alt
        data = data[np.lexsort((data[:, 2], data[:, 1], data[:, 0]))]
        data = np.concatenate((data, np.degrees(np.arctan2(data[:, 3], data[:, 4])).reshape(-1, 1),
                               np.sqrt(data[:, 3]**2 + data[:, 4]**2).reshape(-1, 1)), axis=1)  # Append direction and speed to data
        data[:, 2] = data[:, 2]/ft  # input WindSim requires alt in ft
        data[:, 6] = data[:, 6]/kts  # input WindSim requires spd in kts
        # Find new lat, lon pair values in data
        splitvals = np.hstack(
            (0, np.where(np.diff(data[:, 1], axis=0))[0]+1, len(data)))

        # Construct flattend winddata input for add wind function
        for i in range(len(splitvals) - 1):
            # self.addpointvne(lat, lon, vn, ve, alt)
            lat = data[splitvals[i], 0]
            lon = data[splitvals[i], 1]
            winddata = data[splitvals[i]:splitvals[i+1], [2, 5, 6]].flatten()
            # WindSim.add(self, lat, lon, *winddata)
            # super().add(lat, lon, *winddata) # TODO: change if inherited from WindSim
            self.add(lat, lon, *winddata)
        

        return True, "Wind field update in area [%d, %d], [%d, %d]. " \
            % (self.lat0, self.lat1, self.lon0, self.lon1) \
            + "time: %04d-%02d-%02d %02d:00" \
            % (self.year, self.month, self.day, self.hour)

    @timed_function(name='WINDGFS', dt=3600)
    def update(self):
        if self.autoload:
            self.create(self.lat0, self.lon0, self.lat1, self.lon1)