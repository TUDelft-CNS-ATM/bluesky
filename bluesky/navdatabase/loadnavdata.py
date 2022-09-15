''' Loader functions for navigation data. '''
import pickle

from bluesky import settings
from bluesky.tools import cachefile
from .loadnavdata_txt import loadnavdata_txt, loadthresholds_txt


# Cache versions: increment these to the current date if the source data is updated
# or other reasons why the cache needs to be updated
navdb_version = 'v20170101'

## Default settings
settings.set_variable_defaults(navdata_path='navdata')


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

            wptdata, aptdata, awydata, firdata, codata = loadnavdata_txt()
            rwythresholds = loadthresholds_txt()

            cache.dump(wptdata)
            cache.dump(awydata)
            cache.dump(aptdata)
            cache.dump(firdata)
            cache.dump(codata)
            cache.dump(rwythresholds)

    return wptdata, aptdata, awydata, firdata, codata, rwythresholds
