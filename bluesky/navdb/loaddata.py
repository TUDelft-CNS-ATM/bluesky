import os

try:
    import cPickle as pickle
except ImportError:
    import pickle

from ..settings import data_path, gui

from load_navdata_txt import load_navdata_txt
from load_visuals_txt import load_coastline_txt

if gui == 'qtgl':
    from load_visuals_txt import load_aptsurface_txt
else:
    from load_visuals_txt import pygame_load_rwythresholds



cachedir = data_path + '/cache'

if not os.path.exists(cachedir):
    os.makedirs(cachedir)


def load_coastlines():
    if not os.path.isfile(cachedir + '/coastlines.p'):
        coastvertices, coastindices = load_coastline_txt()
        with open(cachedir + '/coastlines.p', 'wb') as f:
            pickle.dump(coastvertices, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(coastindices, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open(cachedir + '/coastlines.p', 'rb') as f:
            coastvertices = pickle.load(f)
            coastindices  = pickle.load(f)

    return coastvertices, coastindices


def load_aptsurface():
    if not os.path.isfile(cachedir + '/aptsurface.p'):
        vbuf_asphalt, vbuf_concrete, vbuf_runways, vbuf_rwythr, apt_ctr_lat, apt_ctr_lon, \
            apt_indices, rwythresholds = load_aptsurface_txt()
        with open(cachedir + '/aptsurface.p', 'wb') as f:
            pickle.dump(vbuf_asphalt, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(vbuf_concrete, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(vbuf_runways , f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(vbuf_rwythr , f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(apt_ctr_lat  , f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(apt_ctr_lon  , f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(apt_indices  , f, pickle.HIGHEST_PROTOCOL)
        with open(cachedir + '/rwythresholds.p', 'wb') as f:
            pickle.dump(rwythresholds,  f, pickle.HIGHEST_PROTOCOL)
    else:
        with open(cachedir + '/aptsurface.p', 'rb') as f:
            vbuf_asphalt  = pickle.load(f)
            vbuf_concrete = pickle.load(f)
            vbuf_runways  = pickle.load(f)
            vbuf_rwythr   = pickle.load(f)
            apt_ctr_lat   = pickle.load(f)
            apt_ctr_lon   = pickle.load(f)
            apt_indices   = pickle.load(f)
    return vbuf_asphalt, vbuf_concrete, vbuf_runways, vbuf_rwythr, \
        apt_ctr_lat, apt_ctr_lon, apt_indices


def load_navdata():
    
    # Loading of waypoints.data, airports.dat and fir folder from cache or files
      
    data_version  = 1     # Current version nav data, increase to force refresh cache
    cache_ok = False # We still need to check cache content and version

    print "load_navdata: data version",data_version
    
    # Check or regenerate cache
    while not cache_ok:

        # Does cache exist or not? If not, make a new cachefile
        if not os.path.isfile(cachedir + '/navdata.p'):

            wptdata, aptdata, firdata = load_navdata_txt()

            with open(cachedir + '/navdata.p', 'wb') as f:
                pickle.dump(data_version, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(wptdata,      f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(aptdata,      f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(firdata,      f, pickle.HIGHEST_PROTOCOL)

            # Renewed cache so now ok:
            cache_ok = True

        # If it exists, read it and check it
        else:
            
            # Try reading data from cache
            with open(cachedir + '/navdata.p', 'rb') as f:

                cache_version = pickle.load(f)

                # Check whether version number is in data
                if cache_version == data_version:
                    wptdata = pickle.load(f)
                    aptdata = pickle.load(f)
                    firdata = pickle.load(f)

                # Do not read data from an invalid file, set empty dicts
                else:
                    wptdata = {}
                    aptdata = {}
                    firdata = {}

            # Check whether cached data is okay, insert any data check here
            # If it is the same set of data:
            wpdata_ok  = set(wptdata.keys()) == \
                         set(["wpid","wplat","wplon","wpapt","wptype","wpco"])
                                                                             
            apdata_ok  = set(aptdata.keys()) == \
                         set(["apid","apname","aplat","aplon","apmaxrwy","aptype",
                              "apco","apelev"])
                                              
            firdata_ok = set(firdata.keys()) == \
                         set(["fir","firlat0","firlon0","firlat1","firlon1"])

            # If any of data not ok, cache is not ok
            cache_ok = wpdata_ok and apdata_ok and firdata_ok       

            # If not delete cache file to trigger rewriting it
            if not cache_ok:
                os.remove(cachedir + '/navdata.p')
                print "Found an error: Removing old cache-file navdata.p"

    if not os.path.isfile(cachedir + '/rwythresholds.p'):
        if gui == 'qtgl':
            load_aptsurface()
        else:
            rwythresholds = pygame_load_rwythresholds()
            with open(cachedir + '/rwythresholds.p', 'wb') as f:
                pickle.dump(rwythresholds,  f, pickle.HIGHEST_PROTOCOL)

    with open(cachedir + '/rwythresholds.p', 'rb') as f:
        rwythresholds = pickle.load(f)
    return wptdata, aptdata, firdata, rwythresholds
