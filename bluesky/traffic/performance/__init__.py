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
