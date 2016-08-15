from bluesky import settings
if __name__ == "__main__":
    print "   *****   BlueSky Open ATM simulator *****"
    print "Distributed under GNU General Public License v3"
    settings.init('pygame')

from bluesky.navdb import Navdatabase
from bluesky.ui.pygame import Gui
from bluesky.sim.pygame import Simulation


# Global navdb, gui, and sim objects for easy access in interactive python shell
navdb = None
gui   = None
sim   = None


def MainLoop():
    # =============================================================================
    # Create gui and simulation objects
    # =============================================================================
    global navdb, gui, sim
    navdb = Navdatabase('global')   # Read database from specified folder
    gui   = Gui(navdb)
    sim   = Simulation(gui, navdb)

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
    # =============================================================================
    # Clean up before exit. Comment this out when debugging for checking variables
    # in the shell.
    # =============================================================================
    del gui
    #-debug del sim
    print 'BlueSky normal end.'
    return

if __name__ == '__main__':
    # Run mainloop if BlueSky_pygame is called directly
    MainLoop()
