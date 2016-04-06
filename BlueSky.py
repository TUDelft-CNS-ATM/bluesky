import bluesky.settings as settings
settings.init()

if settings.gui == 'pygame':
    import BlueSky_pygame as bs

    # Initialisation
    gui, sim = bs.CreateMainObj()

    # Main loop
    sim = bs.MainLoop(gui,sim)


elif settings.gui == 'qtgl':
    import BlueSky_qtgl as bs

    # Initialisation
    gui, manager, sim = bs.CreateMainObj()

    # Main loop
    sim = bs.MainLoop(gui,manager)

    
else:
    import sys
    print 'Unknown gui type:', settings.gui
    sys.exit(0)


print "Bluesky: Normal end."
