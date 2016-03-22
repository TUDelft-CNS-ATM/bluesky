import bluesky.settings as settings
settings.init()

if settings.gui == 'pygame':
    import BlueSky_pygame as bs

elif settings.gui == 'qtgl':
    import BlueSky_qtgl as bs

else:
    import sys
    print 'Unknown gui type:', settings.gui
    sys.exit(0)

gui, sim = bs.MainLoop()
