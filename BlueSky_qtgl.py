#!/usr/bin/env python
""" OpenGL BlueSky start script """
from __future__ import print_function
import sys
import traceback
import bluesky as bs

if bs.settings.is_gui:
    from bluesky.simulation.qtgl.mainmanager import MainManager
    from bluesky.ui.qtgl import Gui
    from bluesky.tools.network import StackTelnetServer
    if __name__ == "__main__":
        print("   *****   BlueSky Open ATM simulator *****")
        print("Distributed under GNU General Public License v3")


# Global gui object for easy access in interactive python shell
gui = None

# Register settings defaults
bs.settings.set_variable_defaults(telnet_port=8888)

# Create custom system-wide exception handler. For now it replicates python's
# default traceback message. This was added to counter a new PyQt5.5 feature
# where unhandled exceptions would result in a qFatal with a very uninformative
# message.


def exception_handler(exc_type, exc_value, exc_traceback):
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    sys.exit()


sys.excepthook = exception_handler


def start():
    """
    Start BlueSky: Create gui and simulation objects
    """
    global gui
    telnet_in = StackTelnetServer()
    manager = MainManager(telnet_in)
    gui = Gui()

    # Initialize the gui (loading graphics data, etc.)
    gui.init()

    # Connect gui stack command to telnet_in
    telnet_in.connect(gui.win.console.stack)

    # Start the node manager
    manager.start()

    # Start the telnet input server for stack commands
    telnet_in.listen(port=bs.settings.telnet_port)

    return telnet_in, manager


def gui_prestart():
    """
    Set up running of GUI
    """
    gui.prestart()


def gui_exec():
    """
    Execute running of GUI
    """
    gui.exec_()


def stop(telnet_in, manager):
    """
    Tear-down BlueSky
    """
    print('Stopping telnet server.')
    telnet_in.close()

    # Close the manager, stop all nodes
    manager.stop()

    # ======================================================================
    # Clean up before exit. Comment this out when debugging for checking
    # variables in the shell.
    # ======================================================================
    global gui
    del gui
    print('BlueSky normal end.')


# =============================================================================
# Start the mainloop (and possible other threads)
# =============================================================================
def main_loop():
    # Initialize bluesky modules
    bs.init()

    if bs.settings.is_sim:
        from bluesky.simulation.qtgl import nodemanager as manager
        manager.run()

    else:
        telnet_in, manager = start()
        gui.start()
        stop(telnet_in, manager)


if __name__ == "__main__":
    # Run mainloop if BlueSky-qtgl is called directly
    main_loop()
