''' BlueSky functions for geographical calculations. '''
from bluesky import settings
from bluesky.tools.geo import _geo
from bluesky.tools.geo._geo import nm, magdec, initdecl_data, magdeccmd, kwikpos


# Register settings defaults
settings.set_variable_defaults(prefer_compiled=False)


class _Unselected:
    ''' Stands in for the backend until one is selected, and selects it from
        the current settings on first use. '''
    def __getattr__(self, name):
        return getattr(select_backend(), name)


# Backend module (_geo or _cgeo) that implements the functions below
_impl = _Unselected()


def select_backend():
    ''' Select the compiled or the Python implementation of the geo functions,
        depending on settings.prefer_compiled. '''
    global _impl
    _impl = _geo
    if settings.prefer_compiled:
        try:
            from bluesky.tools.geo import _cgeo
            _impl = _cgeo
        except ImportError:
            pass
    return _impl


def init():
    ''' Select the geo backend after settings.cfg has been read. '''
    select_backend()
    if _impl is not _geo:
        print('Using compiled geo functions')
    elif settings.prefer_compiled:
        print('Could not load compiled geo functions, Using Python-based geo functions instead')
    else:
        print('Using Python-based geo functions')


# The functions below forward to the selected backend. They are defined here
# instead of imported from the backend, so that modules which import them
# before bs.init() still use the backend chosen in settings.cfg.
def rwgs84(latd):
    return _impl.rwgs84(latd)


def rwgs84_matrix(latd):
    return _impl.rwgs84_matrix(latd)


def qdrdist(latd1, lond1, latd2, lond2):
    return _impl.qdrdist(latd1, lond1, latd2, lond2)


def qdrdist_matrix(lat1, lon1, lat2, lon2):
    return _impl.qdrdist_matrix(lat1, lon1, lat2, lon2)


def latlondist(latd1, lond1, latd2, lond2):
    return _impl.latlondist(latd1, lond1, latd2, lond2)


def latlondist_matrix(lat1, lon1, lat2, lon2):
    return _impl.latlondist_matrix(lat1, lon1, lat2, lon2)


def wgsg(latd):
    return _impl.wgsg(latd)


def qdrpos(latd1, lond1, qdr, dist):
    return _impl.qdrpos(latd1, lond1, qdr, dist)


def kwikdist(lata, lona, latb, lonb):
    return _impl.kwikdist(lata, lona, latb, lonb)


def kwikdist_matrix(lata, lona, latb, lonb):
    return _impl.kwikdist_matrix(lata, lona, latb, lonb)


def kwikqdrdist(lata, lona, latb, lonb):
    return _impl.kwikqdrdist(lata, lona, latb, lonb)


def kwikqdrdist_matrix(lata, lona, latb, lonb):
    return _impl.kwikqdrdist_matrix(lata, lona, latb, lonb)


for _f in (rwgs84, rwgs84_matrix, qdrdist, qdrdist_matrix, latlondist, latlondist_matrix,
           wgsg, qdrpos, kwikdist, kwikdist_matrix, kwikqdrdist, kwikqdrdist_matrix):
    _f.__doc__ = getattr(_geo, _f.__name__).__doc__
del _f
