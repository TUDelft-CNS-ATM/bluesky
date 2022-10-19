from pathlib import Path
import sys
import pygrib
import datetime
import requests
import numpy as np
import bluesky as bs
from bluesky import stack
from bluesky.core import timed_function
from bluesky.traffic.windsim import WindSim

bs.settings.set_variable_defaults(
    windgfs_url="https://www.ncei.noaa.gov/data/global-forecast-system/access/historical/analysis/")

# nlayer = 23

datadir = Path('')

def init_plugin():
    global datadir
    datadir = bs.resource(bs.settings.data_path) / 'grib'

    if not datadir.is_dir():
        datadir.mkdir()

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
        self.year  = 0
        self.month = 0
        self.day   = 0
        self.hour  = 0
        self.lat0  = -90
        self.lon0  = -180
        self.lat1  = 90
        self.lon1  = 180

        # Switch for periodic loading of new GFS data
        self.autoload = True

    def fetch_grb(self, year, month, day, hour, pred=0):
        ym = "%04d%02d" % (year, month)
        ymd = "%04d%02d%02d" % (year, month, day)
        hm = "%02d00" % hour
        pred = "%03d" % pred

        remote_loc = "/%s/%s/gfsanl_3_%s_%s_%s.grb2" % (ym, ymd, ymd, hm, pred)

        fname = "gfsanl_3_%s_%s_%s.grb2" % (ymd, hm, pred)
        fpath = datadir / fname

        remote_url = bs.settings.windgfs_url + remote_loc

        if not fpath.is_file():
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

            if level < 100:  # lesss than 100 hPa, above about 54 k ft
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
        self.lat0, self.lon0, self.lat1, self.lon1 =  min(lat0, lat1), \
                              min(lon0, lon1), max(lat0, lat1), max(lon0, lon1)
        self.year = year or bs.sim.utc.year
        self.month = month or bs.sim.utc.month
        self.day = day or bs.sim.utc.day
        self.hour = hour or bs.sim.utc.hour

        # round hour to 3 hours, check if it is a +3h prediction
        self.hour = round(self.hour/3) * 3
        if self.hour in [3, 9, 15, 21]:
            self.hour = self.hour - 3
            pred = 3
        elif self.hour == 24:
            ymd0 = "%04d%02d%02d" % (self.year, self.month, self.day)
            print(ymd0)
            ymd1 = (datetime.datetime.strptime(ymd0, '%Y%m%d') + 
                    datetime.timedelta(days=1))
            self.year  = ymd1.year
            self.month = ymd1.month
            self.day   = ymd1.day    
            self.hour  = 0
            pred = 0
        else:
            pred = 0

        txt = "Loading wind field for %s-%s-%s %s:00..." % (self.year, self.month, self.day, self.hour)
        bs.scr.echo("%s" % txt)

        grb = self.fetch_grb(self.year, self.month, self.day, self.hour, pred)

        if grb is None or self.lat0 == self.lat1 or self.lon0 == self.lon1:
            return False, "Wind data non-existend in area [%d, %d], [%d, %d]. " \
                % (self.lat0, self.lat1, self.lon0, self.lon1) \
                + "time: %04d-%02d-%02d %02d:00" \
                % (self.year, self.month, self.day, self.hour)

        # first clear exisiting wind field
        self.clear()

        # add new wind field
        data = self.extract_wind(grb, self.lat0, self.lon0, self.lat1, self.lon1).T

        data = data[np.lexsort((data[:, 2], data[:, 1], data[:, 0]))] # Sort by lat, lon, alt
        reshapefactor = int((1 + max(self.lat0, self.lat1) - min(self.lat0, self.lat1)) * \
                            (1 + max(self.lon0, self.lon1) - min(self.lon0, self.lon1)))

        lat     = np.reshape(data[:,0], (reshapefactor, -1)).T[0,:]
        lon     = np.reshape(data[:,1], (reshapefactor, -1)).T[0,:]
        veast   = np.reshape(data[:,3], (reshapefactor, -1)).T
        vnorth  = np.reshape(data[:,4], (reshapefactor, -1)).T
        windalt = np.reshape(data[:,2], (reshapefactor, -1)).T[:,0]

        self.addpointvne(lat, lon, vnorth, veast, windalt)        

        return True, "Wind field updated in area [%d, %d], [%d, %d]. " \
            % (self.lat0, self.lat1, self.lon0, self.lon1) \
            + "time: %04d-%02d-%02d %02d:00" \
            % (self.year, self.month, self.day, self.hour)

    @timed_function(name='WINDGFS', dt=3600)
    def update(self):
        if self.autoload:
            _, txt = self.loadwind(self.lat0, self.lon0, self.lat1, self.lon1)
            bs.scr.echo("%s" % txt)