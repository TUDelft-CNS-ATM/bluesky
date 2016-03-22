from bluesky import settings
settings.init('pygame')

from bluesky.traf import Navdatabase
from bluesky.ui.pygame import Gui
from bluesky.sim.pygame import Simulation

def MainLoop():
    # =============================================================================
    # Create gui and simulation objects
    # =============================================================================
    navdb = Navdatabase('global')   # Read database from specified folder 
    gui   = Gui(navdb)
    sim   = Simulation(gui,navdb)

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

    return gui, sim

if __name__ == '__main__':
    # Run mainloop if BlueSky-qtgl is called directly
    gui, sim = MainLoop()

    # =============================================================================
    # Clean up before exit. Comment this out when debugging for checking variables
    # in the shell.
    # =============================================================================
    del gui
    #-debug del sim

print 'BlueSky normal end.'
