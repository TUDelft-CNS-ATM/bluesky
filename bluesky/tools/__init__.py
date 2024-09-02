''' BlueSky tools. '''


def init():
    import bluesky.tools.geo as geo
    print("Reading magnetic variation data")
    geo.initdecl_data()
