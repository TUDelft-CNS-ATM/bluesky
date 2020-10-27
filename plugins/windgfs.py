import os
import sys
import numpy as np
import pandas as pd
import pygrib
import requests
import bluesky as bs
from bluesky import settings, stack

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
        'plugin_type': 'sim',
        'update_interval': 3600,
        'update': windgfs.update
    }

    stackfunctions = {
        "WINDGFS": [
            "WINDGFS lat0,lon0,lat1,lon1,[year,month,day,hour]",
            "latlon,latlon,[int,int,int,int]",
            windgfs.create,
            "Select an ADS-B data source for traffic"]}

    return config, stackfunctions

class WindGFS:
    def __init__(self):
        self.year = 0
        self.month = 0
        self.day = 0
        self.hour = 0
        self.lat0 = -90
        self.lon0 = -180
        self.lat1 = 90
        self.lon1 = 180


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

        mask = (lats > lat0_) & (lats < lat1_) & (lons > lon0_) & (lons < lon1_)

        data = np.array([lats[mask], lons[mask], alts[mask], vxs[mask], vys[mask]])

        return data


    def create(self, *args):
        if len(args) == 0:
            pass

        if len(args) == 4:
            self.lat0, self.lon0, self.lat1, self.lon1 =  args
            self.year, self.month, self.day = bs.sim.utc.year, bs.sim.utc.month, bs.sim.utc.day
            self.hour = bs.sim.utc.hour

        elif len(args) == 8:
            self.lat0, self.lon0, self.lat1, self.lon1, \
            self.year, self.month, self.day, self.hour =  args


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
        stack.stack('DEL wind')

        # add new wind field
        data = self.extract_wind(grb, self.lat0, self.lon0, self.lat1, self.lon1)
        df = pd.DataFrame(data.T, columns=['lat','lon','alt','vx','vy'])
        df['dir'] = np.degrees(np.arctan2(df.vx, df.vy))
        df['spd'] = np.sqrt(df.vx**2 + df.vy**2)

        for (lat, lon), d in df.groupby(['lat', 'lon']):
            cmd = "WIND %d,%d," % (lat, lon)
            for idx, r in d.iterrows():
                cmd += "%d,%d,%d," % (r.alt, r.dir, r.spd)
            stack.stack(cmd)

        return True, "Wind field update in area [%d, %d], [%d, %d]. " \
            % (self.lat0, self.lat1, self.lon0, self.lon1) \
            + "time: %04d-%02d-%02d %02d:00" \
            % (self.year, self.month, self.day, self.hour)


    def update(self):
        return self.create(self.lat0, self.lon0, self.lat1, self.lon1)
