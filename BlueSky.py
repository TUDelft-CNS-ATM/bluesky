from bluesky.settings import gui
if gui == 'pygame':
    import BlueSky_pygame as bs
elif gui == 'qtgl':
    import BlueSky_qtgl as bs
else:
    import sys
    print 'Unknown gui type:', gui
    sys.exit(0)

bs.MainLoop()
