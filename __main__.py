#!/usr/bin/env python
# -*- coding: utf8 -*-
""" Overall BlueSky start script """
from __future__ import print_function
from bluesky import settings
import bluesky as bs
import traceback


print("   *****   BlueSky Open ATM simulator *****")
print("Distributed under GNU General Public License v3")

if settings.gui == 'pygame':
    from bluesky.ui.pygame import splash
    import pygame as pg

    def start():
        # =============================================================================
        # Start the mainloop (and possible other threads)
        # =============================================================================
        splash.show()
        bs.init()
        bs.sim.operate()
        bs.scr.init()

        # Main loop for tmx object
        while not bs.sim.mode == bs.sim.end:
            bs.sim.update()   # Update sim
            bs.scr.update()   # GUI update

            # Restart traffic simulation:
            if bs.sim.mode == bs.sim.init:
                bs.sim.reset()
                bs.scr.objdel()     # Delete user defined objects

elif settings.gui == 'qtgl':

    if bs.settings.is_gui or bs.settings.is_headless:
        print("   *****   BlueSky Open ATM simulator *****")
        print("Distributed under GNU General Public License v3")


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
        missingmodules = {
            "OpenGL": "pyopengl-accelerate",
            "PyQt4": "pyqt5"
        }

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
        except (ModuleNotFoundError, ImportError) as error:
            modulename = missingmodules.get(error.name) or error.name
            print("Bluesky needs", modulename, file=sys.stderr)
            print("Install using e.g. pip install {}".format(modulename),
                  file=sys.stderr)

else:
    import sys
    print('Unknown gui type: {}'.format(settings.gui), file=sys.stderr)
    sys.exit(0)

def cleanup():
    # After the simulation is done, close the gui
    if settings.gui == 'pygame':
        bs.sim.stop()
        pg.quit()

    print('BlueSky normal end.')

# Start the main loop. When debugging in python interactive mode,
# relevant objects are available in bs namespace (e.g., bs.scr, bs.sim)
start()

cleanup()