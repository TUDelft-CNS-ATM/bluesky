import os
import cdsapi
import numpy as np
import bluesky as bs
import netCDF4 as nc
from bluesky import settings, stack
from bluesky.core import timed_function
from bluesky.traffic.windsim import WindSim

datadir = settings.data_path + '/NetCDF/'

if not os.path.exists(datadir):
    os.makedirs(datadir)


def init_plugin():
    global windecmwf
    windecmwf = WindECMWF()

    config = {
        'plugin_name': 'WINDECMWF',
        'plugin_type': 'sim'
    }

    return config

class WindECMWF(WindSim):
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
        
        # Switch for pre-fetching data of entire day (due to ECMWF queue)
        self.preload = False
        
    def fetch_nc(self, year, month, day, hour):
        """
        Retrieve weather data via the CDS API for multiple pressure levels
        """
        
        ymd = "%04d%02d%02d" % (year, month, day)
        hm  = "%02d00" % hour
        
        fname = 'p_levels_%s_%s.nc' % (ymd, hm)
        fpath = datadir + fname
        
        if not os.path.isfile(fpath):
            bs.scr.echo("Downloading file, please wait...")
    
            # Set client
            c = cdsapi.Client()
            
            # Retrieve data 
            c.retrieve(
                'reanalysis-era5-pressure-levels',
                {
                    'product_type': 'reanalysis',
                    'format': 'netcdf',
                    'pressure_level': [
                        '125', '150', '175', 
                        '200', '225', '250', 
                        '300', '350', '400', 
                        '450', '500', '550', 
                        '600', '650', '700'
                    ],
                    'year': year,
                    'month': month,
                    'day': day,
                    'time': hour,
                    'variable': [
                        'u_component_of_wind', 'v_component_of_wind'
                    ],
                },
                fpath)
    
        bs.scr.echo("Download completed.")
        netcdf = nc.Dataset(fpath, mode='r')
    
        return netcdf


    def extract_wind(self, netcdf, lat0, lon0, lat1, lon1):

        # Load reanalysis data 
        level = netcdf['level'][:].data
        lats  = netcdf['latitude'][:].data
        lons  = netcdf['longitude'][:].data
        vxs_  = netcdf['u'][:].squeeze().data.flatten()
        vys_  = netcdf['v'][:].squeeze().data.flatten()
        
        # Close data for performance
        netcdf.close()   
        
        # Transform pressure levels to altitude
        p = level * 100
        h = (1 - (p / 101325.0)**0.190264) * 44330.76923    # in meters
        
        # Construct 2D array of all data points
        lats_ = np.tile(np.repeat(lats, len(lons)), len(level))
        lons_ = np.tile(lons, len(lats)*len(level))
        alts_ = np.repeat(h, len(lats)*len(lons))       
     
        # Convert longitudes
        lons_ = (lons_ + 180) % 360.0 - 180.0     # convert range from 0~360 to -180~180
        
        # Reduce area based on lat lon limits
        lat0_ = min(lat0, lat1)
        lat1_ = max(lat0, lat1)
        lon0_ = min(lon0, lon1)
        lon1_ = max(lon0, lon1)

        mask = (lats_ >= lat0_) & (lats_ <= lat1_) & (lons_ >= lon0_) & (lons_ <= lon1_)

        data = np.array([lats_[mask], lons_[mask], alts_[mask], vxs_[mask], vys_[mask]])

        return data

    @stack.command(name='WINDECMWF')
    def loadwind(self, lat0: 'lat', lon0: 'lon', lat1: 'lat', lon1: 'lon',
               year: int=None, month: int=None, day: int=None, hour: int=None):
        ''' WINDECMWF: Load a windfield directly from NOAA database.

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

        # Preload netcdf data
        if self.preload:
            for i in np.arange(0, 24, 3):
                self.fetch_nc(self.year, self.month, self.day, i)

        # round hour to 3 hours, check if it is a +3h prediction
        self.hour  = round(self.hour / 3) * 3

        txt = "Loading wind field for %s-%s-%s %s:00..." % (self.year, self.month, self.day, self.hour)
        bs.scr.echo("%s" % txt)

        netcdf = self.fetch_nc(self.year, self.month, self.day, self.hour)

        if netcdf is None or self.lat0 == self.lat1 or self.lon0 == self.lon1:
            return False, "Wind data non-existend in area [%d, %d], [%d, %d]. " \
                % (self.lat0, self.lat1, self.lon0, self.lon1) \
                + "time: %04d-%02d-%02d %02d:00" \
                % (self.year, self.month, self.day, self.hour)

        # first clear exisiting wind field
        self.clear()

        # add new wind field
        data = self.extract_wind(netcdf, self.lat0, self.lon0, self.lat1, self.lon1).T

        data = data[np.lexsort((data[:, 2], data[:, 1], data[:, 0]))] # Sort by lat, lon, alt
        reshapefactor = int((1 + (max(self.lat0, self.lat1) - min(self.lat0, self.lat1))*4) * \
                            (1 + (max(self.lon0, self.lon1) - min(self.lon0, self.lon1))*4))

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

    @timed_function(name='WINDECMWF', dt=3600)
    def update(self):
        if self.autoload:
            _, txt = self.loadwind(self.lat0, self.lon0, self.lat1, self.lon1)
            bs.scr.echo("%s" % txt)