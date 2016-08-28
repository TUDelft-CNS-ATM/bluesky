import sys
import traceback
from bluesky import settings

if __name__ == "__main__":
    settings.init('qtgl')
 
# This file is used to start the gui mainloop or a single node simulation loop
node_only = ('--node' in sys.argv)

if node_only:
    from bluesky.sim.qtgl.nodemanager import runNode
else:
    from bluesky.navdb import Navdatabase
    from bluesky.ui.qtgl import Gui
    from bluesky.sim.qtgl import MainManager
    from bluesky.tools.network import StackTelnetServer
    if __name__ == "__main__":
        print "   *****   BlueSky Open ATM simulator *****"
        print "Distributed under GNU General Public License v3"


# Global navdb, gui, and sim objects for easy access in interactive python shell
navdb   = None
gui     = None
manager = None


# Create custom system-wide exception handler. For now it replicates python's
# default traceback message. This was added to counter a new PyQt5.5 feature
# where unhandled exceptions would result in a qFatal with a very uninformative
# message.
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
        # ======================================================================
        # Create gui and simulation objects
        # ======================================================================
        global navdb, manager, gui
        manager   = MainManager()
        gui       = Gui()
        navdb     = Navdatabase('global')  # Read database from specified folder
        telnet_in = StackTelnetServer()

        # Initialize the gui (loading graphics data, etc.)
        gui.init(navdb)

        # Start the node manager
        manager.start()

        # Start the telnet input server for stack commands
        telnet_in.listen(port=settings.telnet_port)

        # Start the gui
        gui.start()

        print 'Stopping telnet server.'
        telnet_in.close()

        # Close the manager, stop all nodes
        manager.stop()

        # ======================================================================
        # Clean up before exit. Comment this out when debugging for checking
        # variables in the shell.
        # ======================================================================
        del gui
        print 'BlueSky normal end.'

if __name__ == "__main__":
    # Run mainloop if BlueSky-qtgl is called directly
    MainLoop()
