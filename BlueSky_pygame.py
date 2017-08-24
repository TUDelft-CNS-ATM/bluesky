#!/usr/bin/env python
""" Pygame BlueSky start script """
from __future__ import print_function
import pygame as pg
from bluesky import sim, scr, stack
from bluesky.ui.pygame import Keyboard, splash
from bluesky.tools import plugin
if __name__ == "__main__":
    print("   *****   BlueSky Open ATM simulator *****")
    print("Distributed under GNU General Public License v3")


def main_loop():
    # =============================================================================
    # Start the mainloop (and possible other threads)
    # =============================================================================
    splash.show()
    plugin.init()
    stack.init()
    sim.start()
    scr.init()

    # Main loop for tmx object
    while not sim.mode == sim.end:
        sim.update()   # Update sim
        scr.update()   # GUI update

        # Restart traffic simulation:
        if sim.mode == sim.init:
            sim.reset()
            scr.objdel()     # Delete user defined objects

    # After the simulation is done, close the gui
    sim.stop()
    pg.quit()
    print('BlueSky normal end.')
    return

#==============================================================================

# Run mainloop if BlueSky_pygame is called directly

if __name__ == '__main__':
    main_loop()
