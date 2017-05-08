import sys
import traceback
from bluesky import settings

if __name__ == "__main__":
    settings.init('qtgl')

if not settings.node_only:
    from bluesky.sim.qtgl.mainmanager import MainManager
    from bluesky.ui.qtgl import Gui
    from bluesky.tools.network import StackTelnetServer
    if __name__ == "__main__":
        print "   *****   BlueSky Open ATM simulator *****"
        print "Distributed under GNU General Public License v3"


# Global gui object for easy access in interactive python shell
gui     = None


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
    if settings.node_only:
        import bluesky.sim.qtgl.nodemanager as manager
        manager.run()

    else:
        # ======================================================================
        # Create gui and simulation objects
        # ======================================================================
        global gui
        manager   = MainManager()
        gui       = Gui()
        telnet_in = StackTelnetServer()

        # Initialize the gui (loading graphics data, etc.)
        gui.init()

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
