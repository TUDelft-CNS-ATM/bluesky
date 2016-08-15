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
            vbuf_rwythr  = pickle.load(f)
            apt_ctr_lat   = pickle.load(f)
            apt_ctr_lon   = pickle.load(f)
            apt_indices   = pickle.load(f)
    return vbuf_asphalt, vbuf_concrete, vbuf_runways, vbuf_rwythr, \
        apt_ctr_lat, apt_ctr_lon, apt_indices


def load_navdata():
    if not os.path.isfile(cachedir + '/navdata.p'):
        wptdata, aptdata, firdata = load_navdata_txt()
        with open(cachedir + '/navdata.p', 'wb') as f:
            pickle.dump(wptdata, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(aptdata, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(firdata, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open(cachedir + '/navdata.p', 'rb') as f:
            wptdata = pickle.load(f)
            aptdata = pickle.load(f)
            firdata = pickle.load(f)

    if not gui == 'qtgl':
        # Threshold data is not available for the PyGame version of BlueSky
        return wptdata, aptdata, firdata, []

    if not os.path.isfile(cachedir + '/rwythresholds.p'):
        load_aptsurface()
    with open(cachedir + '/rwythresholds.p', 'rb') as f:
        rwythresholds = pickle.load(f)
    return wptdata, aptdata, firdata, rwythresholds
