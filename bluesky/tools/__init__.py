from bluesky import settings
# Register settings defaults
settings.set_variable_defaults(prefer_compiled=False)
from .walltime import Timer
from .trafficarrays import RegisterElementParameters, TrafficArrays
from .signal import Signal
if settings.prefer_compiled:
    try:
        from . import cgeo as geo
        print('Using compiled geo functions')
    except ImportError:
        from . import geo
        print('Using Python-based geo functions')
else:
    from . import geo
    print('Using Python-based geo functions')

from .trafficarrays import RegisterElementParameters, TrafficArrays
from . import cachefile
