from bluesky import settings
if __name__ == "__main__":
    settings.init('pygame')

from bluesky.traf import Navdatabase
from bluesky.ui.pygame import Gui
from bluesky.sim.pygame import Simulation


def CreateMainObj():
    # =============================================================================
    # Create gui and simulation objects
    # =============================================================================
    navdb = Navdatabase('global')   # Read database from specified folder 
    gui   = Gui(navdb)
    sim   = Simulation(gui,navdb)

    return gui,sim


def MainLoop(gui,sim):

    # =============================================================================
    # Start the mainloop (and possible other threads)
    # =============================================================================
    sim.start()

    # Main loop for tmx object
    while not sim.mode == sim.end:
        sim.update(gui.scr)  # Update sim
        gui.update(sim)      # Update GUI

        # Restart traffic simulation:
        if sim.mode == sim.init:
            sim.reset()
            gui.reset()

    # After the simulation is done, close the gui
    sim.stop()
    gui.close()

    return 

if __name__ == '__main__':
    # Run mainloop if BlueSky-qtgl is called directly
    gui, sim = CreateMainObj()
    
    MainLoop(gui,sim)

    # =============================================================================
    # Clean up before exit. Comment this out when debugging for checking variables
    # in the shell.
    # =============================================================================
    del gui
    #-debug del sim

print 'BlueSky normal end.'
