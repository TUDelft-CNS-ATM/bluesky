import pygame as pg

import bluesky as bs

def show():
    # Show splash screen
    pg.init()
    imgpath = bs.resource("graphics/splash.gif")
    splashimg = pg.image.load(imgpath)
    splashwin = pg.display.set_mode(splashimg.get_size(),pg.NOFRAME)
    splashwin.blit(splashimg,(0,0))
    pg.display.flip()
    return

def destroy():
    pg.display.quit()               # Kill splash screen
    return
