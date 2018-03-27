#!/usr/bin/env python
""" OpenGL BlueSky start script """
from __future__ import print_function
import sys
import traceback
import bluesky as bs


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
    # When importerror gives different name than (pip) install needs, also advise latest version
    missingmodules = {"OpenGL": "pyopengl-accelerate", "PyQt4": "pyqt5"}

    # Catch import errors
    try:
        # Initialize bluesky modules
        bs.init()

        # Start gui if this is the main process
        if bs.settings.is_gui:
            from bluesky.ui import qtgl
            qtgl.start()

        elif bs.settings.is_sim:
            bs.sim.start()

    # Give info on missing module
    except ImportError as error:
        modulename = missingmodules.get(error.name) or error.name
        print("Bluesky needs", modulename)
        print("Install using e.g. pip install", modulename)


def cleanup():
    """
    Tear-down BlueSky
    """
    print('BlueSky normal end.')


if __name__ == "__main__":
    if bs.settings.is_gui or bs.settings.is_headless:
        print("   *****   BlueSky Open ATM simulator *****")
        print("Distributed under GNU General Public License v3")
    # Run mainloop if BlueSky-qtgl is called directly
    start()

    # Cleanup after returning from start()
    cleanup()
