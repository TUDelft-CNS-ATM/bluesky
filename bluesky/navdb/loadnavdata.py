from os import path, listdir

try:
    import cPickle as pickle
except ImportError:
    import pickle

from bluesky import settings

from load_navdata_txt import load_navdata_txt
from load_visuals_txt import load_coastline_txt

if settings.gui == 'qtgl':
    from load_visuals_txt import load_aptsurface_txt
else:
    from load_visuals_txt import pygame_load_rwythresholds

## Default settings
settings.set_variable_defaults(navdata_path='data/navdata', cache_path='data/cache')

sourcedir = settings.navdata_path
cachedir  = settings.cache_path


def check_cache(cachefile, *sources):
    if not path.isfile(cachefile):
        return False
    cachetm = path.getmtime(cachefile)
    for source in sources:
        if path.isfile(source) and path.getmtime(source) > cachetm:
            return False
    return True

def load_coastlines():
    # Check whether anything changed which requires rewriting the cache
    cachefile = path.join(cachedir, 'coastlines.p')
    cache_ok = check_cache(cachefile, path.join(sourcedir, 'coastlines.dat'),
                           'bluesky/navdb/load_visuals_txt.py')

    # If cache up to date, use it
    if cache_ok:
        with open(cachefile, 'rb') as f:
            print "Reading cache: coastlines.p"
            coastvertices = pickle.load(f)
            coastindices  = pickle.load(f)

    # else read original file, and write new cache file
    else:
        coastvertices, coastindices = load_coastline_txt()
        with open(cachefile, 'wb') as f:
            print "Writing cache: coastlines.p"
            pickle.dump(coastvertices, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(coastindices, f, pickle.HIGHEST_PROTOCOL)

    return coastvertices, coastindices


def load_aptsurface():
    # Check whether anything changed which requires rewriting the cache
    cachefile = path.join(cachedir, 'aptsurface.p')
    cache_ok = check_cache(cachefile, path.join(sourcedir, 'apt.zip'),
                           'bluesky/navdb/load_visuals_txt.py')

    # If cache up to date, use it
    if cache_ok:
        with open(cachefile, 'rb') as f:
            print "Reading cache: aptsurface.p"
            vbuf_asphalt  = pickle.load(f)
            vbuf_concrete = pickle.load(f)
            vbuf_runways  = pickle.load(f)
            vbuf_rwythr   = pickle.load(f)
            apt_ctr_lat   = pickle.load(f)
            apt_ctr_lon   = pickle.load(f)
            apt_indices   = pickle.load(f)
            rwythresholds = pickle.load(f)

    # else read original files, and write new cache file
    else:
        print cachefile
        vbuf_asphalt, vbuf_concrete, vbuf_runways, vbuf_rwythr, apt_ctr_lat, apt_ctr_lon, \
            apt_indices, rwythresholds = load_aptsurface_txt()
        with open(cachefile, 'wb') as f:
            print "Writing cache: aptsurface.p"
            pickle.dump(vbuf_asphalt, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(vbuf_concrete, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(vbuf_runways , f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(vbuf_rwythr , f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(apt_ctr_lat  , f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(apt_ctr_lon  , f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(apt_indices  , f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(rwythresholds, f, pickle.HIGHEST_PROTOCOL)

    return vbuf_asphalt, vbuf_concrete, vbuf_runways, vbuf_rwythr, \
        apt_ctr_lat, apt_ctr_lon, apt_indices, rwythresholds


def load_navdata():
    # Check whether anything changed which requires rewriting the cache
    cachefile = path.join(cachedir, 'navdata.p')
    sources   = [path.join(sourcedir, f + '.dat') for f in ['nav', 'fix', 'awy', 'airports', 'icao-countries']]
    sources  += [path.join(sourcedir, 'fir', f) for f in listdir(path.join(sourcedir, 'fir'))]
    sources  += 'bluesky/navdb/load_navdata_txt.py'
    cache_ok  = check_cache(cachefile, *sources)

    if not cache_ok:
        wptdata, aptdata, awydata, firdata, codata = load_navdata_txt()
        if settings.gui == 'qtgl':
            vbuf_asphalt, vbuf_concrete, vbuf_runways, vbuf_rwythr, \
                apt_ctr_lat, apt_ctr_lon, apt_indices, rwythresholds = load_aptsurface()

        else:
            rwythresholds = pygame_load_rwythresholds()

        with open(path.join(cachedir, 'navdata.p'), 'wb') as f:
            print "Writing cache: navdata.p"
            pickle.dump(wptdata,       f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(awydata,       f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(aptdata,       f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(firdata,       f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(codata,        f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(rwythresholds, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open(path.join(cachedir, 'navdata.p'), 'rb') as f:
            print "Reading cache: navdata.p"
            wptdata       = pickle.load(f)
            awydata       = pickle.load(f)
            aptdata       = pickle.load(f)
            firdata       = pickle.load(f)
            codata        = pickle.load(f)
            rwythresholds = pickle.load(f)

    return wptdata, aptdata, awydata, firdata, codata, rwythresholds
