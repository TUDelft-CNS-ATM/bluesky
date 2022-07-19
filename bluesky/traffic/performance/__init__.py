from bluesky import settings
from bluesky.traffic.performance.perfbase import PerfBase
try:
    import bluesky.traffic.performance.openap
    print('Successfully loaded OpenAP performance model')
except ImportError:
    print('Failed to load OpenAP performance model')
try:
    import bluesky.traffic.performance.bada
    print('Successfully loaded BADA performance model')
except ImportError:
    print('Failed to load BADA performance model')
try:
    import bluesky.traffic.performance.legacy
    print('Successfully loaded legacy performance model')
except ImportError:
    print('Failed to load BlueSky legacy performance model')

settings.set_variable_defaults(performance_model='openap')

# Set default performance model
PerfBase.setdefault(settings.performance_model)
