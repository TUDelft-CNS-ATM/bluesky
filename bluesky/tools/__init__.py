''' BlueSky tools. '''


def init():
    import bluesky.tools.geo as geo
    geo.init()
    print("Reading magnetic variation data")
    geo.initdecl_data()
