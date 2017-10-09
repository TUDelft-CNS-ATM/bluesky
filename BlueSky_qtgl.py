#!/usr/bin/env python
""" OpenGL BlueSky start script """
from __future__ import print_function
import sys
import traceback
import bluesky as bs

if bs.settings.is_gui and __name__ == "__main__":
    print("   *****   BlueSky Open ATM simulator *****")
    print("Distributed under GNU General Public License v3")

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
    # Initialize bluesky modules, and start manager
    bs.init()
    bs.manager.start()

    # Start gui if this is the main process
    if bs.settings.is_gui:
        from bluesky.ui import qtgl
        qtgl.start()


def cleanup():
    """
    Tear-down BlueSky
    """
    # Close the manager, stop all nodes
    bs.manager.stop()
    print('BlueSky normal end.')


if __name__ == "__main__":
    # Run mainloop if BlueSky-qtgl is called directly
    start()

    # Cleanup after returning from start()
    cleanup()

    print('BlueSky normal end.')
