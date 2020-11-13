from bluesky import settings
from bluesky.traffic.performance.perfbase import PerfBase
try:
    import bluesky.traffic.performance.openap
except ImportError:
    print('Failed to load OpenAP performance model')
try:
    import bluesky.traffic.performance.bada
except ImportError:
    print('Failed to load BADA performance model')
try:
    import bluesky.traffic.performance.legacy
except ImportError:
    print('Failed to load BlueSky legacy performance model')

settings.set_variable_defaults(performance_model='openap')

# Set default performance model
PerfBase.setdefault(settings.performance_model)
