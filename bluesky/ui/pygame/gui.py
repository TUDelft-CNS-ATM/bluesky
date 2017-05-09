from keyboard import Keyboard
import bluesky as bs
import splash
import pygame as pg


class Gui:
    def __init__(self):
        splash.show()
        self.keyb = Keyboard()                      # processes input from keyboard & mouse
        bs.scr.updateNavBuffers()

    def update(self):
        self.keyb.update()  # Check for keys & mouse
        bs.scr.update()     # GUI update

    def reset(self):
        bs.scr.objdel()     # Delete user defined objects

    def close(self):
        pg.quit()
