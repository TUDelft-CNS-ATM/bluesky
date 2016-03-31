import sys
import traceback
from bluesky import settings

if __name__ == "__main__":
    settings.init('qtgl')

# This file can be used to start the gui mainloop or a single node simulation loop
node_only = ('--node' in sys.argv)

if node_only:
    from bluesky.sim.qtgl.nodemanager import runNode
else:
    from bluesky.traf import Navdatabase
    from bluesky.ui.qtgl import Gui
    from bluesky.sim.qtgl import MainManager


# Global navdb, gui, and sim objects for easy access in interactive python shell
navdb   = None
gui     = None
manager = None


# Create custom system-wide exception handler. For now it replicates python's default traceback message.
# This was added to counter a new PyQt5.5 feature where unhandled exceptions would result in a qFatal
# with a very uninformative message
def exception_handler(exc_type, exc_value, exc_traceback):
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    sys.exit()

sys.excepthook = exception_handler


# =============================================================================
# Start the mainloop (and possible other threads)
# =============================================================================
def MainLoop():
    if node_only:
        runNode()

    else:
        # =============================================================================
        # Create gui and simulation objects
        # =============================================================================
        global navdb, manager, gui
        navdb     = Navdatabase('global')  # Read database from specified folder
        manager   = MainManager(navdb)
        gui       = Gui(navdb)

        # Start the node manager
        manager.start()

        # Start the gui
        gui.start()

if __name__ == "__main__":
    # Run mainloop if BlueSky-qtgl is called directly
    MainLoop()

    # =============================================================================
    # Clean up before exit. Comment this out when debugging for checking variables
    # in the shell.
    # =============================================================================
    del gui
    #-debug del sim
    print 'BlueSky normal end.'
