import pygame as pg
from bluesky import settings, sim, scr, stack
from bluesky.ui.pygame import Keyboard, splash
if __name__ == "__main__":
    print "   *****   BlueSky Open ATM simulator *****"
    print "Distributed under GNU General Public License v3"
    settings.init('pygame')


def MainLoop():
    # =============================================================================
    # Start the mainloop (and possible other threads)
    # =============================================================================
    keyb = Keyboard()                      # processes input from keyboard & mouse
    splash.show()
    scr.updateNavBuffers()
    stack.init()
    sim.start()

    # Main loop for tmx object
    while not sim.mode == sim.end:
        sim.update()   # Update sim
        keyb.update()  # Check for keys & mouse
        scr.update()   # GUI update

        # Restart traffic simulation:
        if sim.mode == sim.init:
            sim.reset()
            scr.objdel()     # Delete user defined objects

    # After the simulation is done, close the gui
    sim.stop()
    pg.quit()
    print 'BlueSky normal end.'
    return

#==============================================================================

# Run mainloop if BlueSky_pygame is called directly

if __name__ == '__main__':
    MainLoop()
