from keyboard import Keyboard
from screen import Screen
import splash
import pygame as pg


class Gui:
    def __init__(self, navdb):
        splash.show()
        self.keyb = Keyboard()                                # processes input from keyboard & mouse
        self.scr  = Screen()                                  # screen output object
        self.scr.updateNavBuffers(navdb)

    def update(self, sim):
        self.keyb.update(sim, sim.stack, self.scr, sim.traf)  # Check for keys & mouse
        self.scr.update(sim, sim.traf)                        # GUI update

    def reset(self):
        self.scr.objdel()                                     # Delete user defined objects

    def close(self):
        pg.quit()
