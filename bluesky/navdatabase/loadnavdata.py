''' Loader functions for navigation data. '''
import pickle

import bluesky as bs
from bluesky import settings
from bluesky.tools import cachefile
from .load_navdata_txt import load_navdata_txt
from .load_visuals_txt import load_coastline_txt, navdata_load_rwythresholds

# Only try this if BlueSky is started in qtgl gui mode
if bs.gui_type == 'qtgl':
    from .load_visuals_txt import load_aptsurface_txt


# Cache versions: increment these to the current date if the source data is updated
# or other reasons why the cache needs to be updated
coast_version = 'v20170101'
navdb_version = 'v20170101'
aptsurf_version = 'v20171116'

## Default settings
settings.set_variable_defaults(navdata_path='data/navdata')

sourcedir = settings.navdata_path

def load_coastlines():
    ''' Load coastline data for gui. '''
    with cachefile.openfile('coastlines.p', coast_version) as cache:
        try:
            coastvertices = cache.load()
            coastindices = cache.load()
        except (pickle.PickleError, cachefile.CacheError) as e:
            print(e.args[0])
            coastvertices, coastindices = load_coastline_txt()
            cache.dump(coastvertices)
            cache.dump(coastindices)

    return coastvertices, coastindices


def load_aptsurface():
    ''' Load airport surface polygons for gui. '''
    with cachefile.openfile('aptsurface.p', aptsurf_version) as cache:
        try:
            vbuf_asphalt  = cache.load()
            vbuf_concrete = cache.load()
            vbuf_runways  = cache.load()
            vbuf_rwythr   = cache.load()
            apt_ctr_lat   = cache.load()
            apt_ctr_lon   = cache.load()
            apt_indices   = cache.load()
        except (pickle.PickleError, cachefile.CacheError) as e:
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
    with cachefile.openfile('navdata.p', navdb_version) as cache:
        try:
            wptdata       = cache.load()
            awydata       = cache.load()
            aptdata       = cache.load()
            firdata       = cache.load()
            codata        = cache.load()
            rwythresholds = cache.load()
        except (pickle.PickleError, cachefile.CacheError) as e:
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
