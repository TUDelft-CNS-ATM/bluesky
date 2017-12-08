''' Loader functions for navigation data. '''
from os import path
try:
    import cPickle as pickle
except ImportError:
    import pickle

from bluesky import settings

from .load_navdata_txt import load_navdata_txt
from .load_visuals_txt import load_coastline_txt, navdata_load_rwythresholds

if settings.gui == 'qtgl':
    from .load_visuals_txt import load_aptsurface_txt


# Cache versions: increment these to the current date if the source data is updated
# or other reasons why the cache needs to be updated
coast_version = 'v20170101'
navdb_version = 'v20170101'
aptsurf_version = 'v20171205'

## Default settings
settings.set_variable_defaults(navdata_path='data/navdata', cache_path='data/cache')

sourcedir = settings.navdata_path
cachedir  = settings.cache_path

class CacheError(Exception):
    ''' Exception class for CacheFile errors. '''
    pass


class CacheFile():
    ''' Convenience class for loading and saving pickle cache files. '''
    def __init__(self, fname, version_ref):
        self.fname = fname
        self.version_ref = version_ref
        self.file = None

    def check_cache(self):
        ''' Check whether the cachefile exists, and is of the correct version. '''
        if not path.isfile(self.fname):
            raise CacheError('Cachefile not found: ' + self.fname)

        self.file = open(self.fname, 'rb')
        version = pickle.load(self.file)

        # Version check
        if not version == self.version_ref:
            self.file.close()
            self.file = None
            raise CacheError('Cache file out of date: ' + self.fname)
        print('Reading cache: ' + self.fname)

    def load(self):
        ''' Load a variable from the cache file. '''
        if self.file is None:
            self.check_cache()

        return pickle.load(self.file)

    def dump(self, var):
        ''' Dump a variable to the cache file. '''
        if self.file is None:
            self.file = open(self.fname, 'wb')
            pickle.dump(self.version_ref, self.file, pickle.HIGHEST_PROTOCOL)
            print("Writing cache: " + self.fname)
        pickle.dump(var, self.file, pickle.HIGHEST_PROTOCOL)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()

def load_coastlines():
    ''' Load coastline data for gui. '''
    with CacheFile(path.join(cachedir, 'coastlines.p'), coast_version) as cache:
        try:
            coastvertices = cache.load()
            coastindices = cache.load()
        except (pickle.PickleError, CacheError) as e:
            print(e.args[0])
            coastvertices, coastindices = load_coastline_txt()
            cache.dump(coastvertices)
            cache.dump(coastindices)

    return coastvertices, coastindices


def load_aptsurface():
    ''' Load airport surface polygons for gui. '''
    with CacheFile(path.join(cachedir, 'aptsurface.p'), aptsurf_version) as cache:
        try:
            vbuf_asphalt  = cache.load()
            vbuf_concrete = cache.load()
            vbuf_runways  = cache.load()
            vbuf_rwythr   = cache.load()
            apt_ctr_lat   = cache.load()
            apt_ctr_lon   = cache.load()
            apt_indices   = cache.load()
        except (pickle.PickleError, CacheError) as e:
            print(e.args[0])
            vbuf_asphalt, vbuf_concrete, vbuf_runways, vbuf_rwythr, apt_ctr_lat, \
            apt_ctr_lon, apt_indices = load_aptsurface_txt()
            cache.dump(vbuf_asphalt)
            cache.dump(vbuf_concrete)
            cache.dump(vbuf_runways)
            cache.dump(vbuf_rwythr)
            cache.dump(apt_ctr_lat)
            cache.dump(apt_ctr_lon)
            cache.dump(apt_indices)

    return vbuf_asphalt, vbuf_concrete, vbuf_runways, vbuf_rwythr, \
        apt_ctr_lat, apt_ctr_lon, apt_indices


def load_navdata():
    ''' Load navigation database. '''
    with CacheFile(path.join(cachedir, 'navdata.p'), navdb_version) as cache:
        try:
            wptdata       = cache.load()
            awydata       = cache.load()
            aptdata       = cache.load()
            firdata       = cache.load()
            codata        = cache.load()
            rwythresholds = cache.load()
        except (pickle.PickleError, CacheError) as e:
            print(e.args[0])

            wptdata, aptdata, awydata, firdata, codata = load_navdata_txt()
            rwythresholds = navdata_load_rwythresholds()

            cache.dump(wptdata)
            cache.dump(awydata)
            cache.dump(aptdata)
            cache.dump(firdata)
            cache.dump(codata)
            cache.dump(rwythresholds)

    return wptdata, aptdata, awydata, firdata, codata, rwythresholds
