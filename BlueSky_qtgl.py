from bluesky import settings

settings.init('qtgl')

from bluesky.traf import Navdatabase
from bluesky.ui.qtgl import Gui
from bluesky.sim.qtgl import SimulationManager

# =============================================================================
# Start the mainloop (and possible other threads)
# =============================================================================
def CreateMainObj():
    # =============================================================================
    # Create gui and simulation objects
    # =============================================================================
    navdb     = Navdatabase('global')  # Read database from specified folder
    manager   = SimulationManager(navdb)
    gui       = Gui(navdb)

    # Create the main simulation thread
    manager.addNode()
    return gui, manager, manager.getSimObjectList()
    

def MainLoop(gui,manager):

    # Start the gui
    gui.start()

    # Stop simulation threads
    manager.stop()

    return manager.getSimObjectList()

if __name__ == "__main__":
    # Run mainloop if BlueSky-qtgl is called directly
    gui, manager,sim = CreateMainObj()
    
    sim = MainLoop(gui,manager)

    # =============================================================================
    # Clean up before exit. Comment this out when debugging for checking variables
    # in the shell.
    # =============================================================================
    del gui
    #-debug del sim
    print 'BlueSky normal end.'


