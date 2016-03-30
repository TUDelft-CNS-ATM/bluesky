from bluesky import settings

if __name__ == "__main__":
    settings.init('qtgl')

from bluesky.traf import Navdatabase
from bluesky.ui.qtgl import Gui
from bluesky.sim.qtgl import MainManager

import sys
import traceback


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
def CreateMainObj():
    # =============================================================================
    # Create gui and simulation objects
    # =============================================================================
    navdb     = Navdatabase('global')  # Read database from specified folder
    manager   = MainManager(navdb)
    gui       = Gui(navdb)

    # Create the main simulation thread
    manager.addNode()
    return gui, manager


def MainLoop(gui, manager):

    # Start the node manager
    manager.start()

    # Start the gui
    gui.start()

if __name__ == "__main__":
    # Run mainloop if BlueSky-qtgl is called directly
    gui, manager = CreateMainObj()

    MainLoop(gui, manager)

    # =============================================================================
    # Clean up before exit. Comment this out when debugging for checking variables
    # in the shell.
    # =============================================================================
    del gui
    #-debug del sim
    print 'BlueSky normal end.'
