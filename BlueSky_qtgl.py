from bluesky import settings
settings.init('qtgl')

from bluesky.traf import Navdatabase
from bluesky.ui.qtgl import Gui
from bluesky.sim.qtgl import SimulationManager


# =============================================================================
# Start the mainloop (and possible other threads)
# =============================================================================
def MainLoop():
    # =============================================================================
    # Create gui and simulation objects
    # =============================================================================
    navdb     = Navdatabase('global')  # Read database from specified folder
    manager   = SimulationManager(navdb)
    gui       = Gui(navdb)

    # Create the main simulation thread
    manager.addNode()

    # Start the gui
    gui.start()

    # Stopping simulation thread
    manager.quit()

    return gui, manager.getSimObjectList()


if __name__ == "__main__":
    # Run mainloop if BlueSky-qtgl is called directly
    gui, sim = MainLoop()

    # =============================================================================
    # Clean up before exit. Comment this out when debugging for checking variables
    # in the shell.
    # =============================================================================
    del gui
    #-debug del sim
    print 'BlueSky normal end.'
