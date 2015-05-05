"""
Main module of BlueSky Open ATM Simulator

Creates a tmx object, runs it and deletes it 
(find this main part at the end of this file)

Start this module to start the program

Created by  : Jacco M. Hoekstra (TU Delft)
"""

from bluesky.traf import Traffic
from bluesky.sim import Simulation
from bluesky.ui import Screen, Keyboard
from bluesky.stack import Commandstack
from bluesky.tools import splash

# Only used for debugging
import time
import numpy as np
import pygame as pg   

class TMX():
    """Main BlueSky Traffic simulation main executable (tmx) object"""

    def __init__(self):   #TMX constructor


        # Show splash screen & version number in console
        splash.show()
        print "*** BlueSky Open Air Traffic sim v0.7 ***"

        # Initialize
        print "Initializing Blue Sky objects...."

        # Create objects for simulation, traffic, screen, keyboard & command stack
        # To be used as global

        self.sim  = Simulation(self) # contains time, simmode, etc.
        self.traf = Traffic(self)       # contains data on aircraft
        self.keyb = Keyboard(self)     # processes input from keyboard & mouse 
        self.scr  = Screen(self)         # screen output object
        self.cmd  = Commandstack(self)    # list with strings with commands from file or user
        return
        

    def run(self):
        """Start running the simulator (set sim mode)"""
        self.sim.start()

        # Main loop for tmx object
        while not self.sim.mode == self.sim.end :
            self.sim.update()                    # Update clock: t, dt
            self.keyb.update()                   # Check for keys & mouse
            self.cmd.checkfile()                 # Process input file
            self.cmd.process()                   # Process commands
            self.traf.update()                   # Traffic movements and controls
            self.scr.update()                    # GUI update

            # Restart traffic simulation:
            if self.sim.mode == self.sim.init:
                self.reset()
        return 


    def reset(self):
        """Reset Traffic database"""
        del self.traf
        self.traf = Traffic(self)
        self.sim.start()
        self.scr.objdel()              # Delete user defined objects
        return        

    def stack(self,txt):
        """Add a commandline to stack, may be multiple commands separated by"""
        self.cmd.stack(txt)        
        return
        

    def __del__(self):
        """Clean up. tmx object destructor, called when object is deleted """
        print "Deleting BlueSky objects..."
        del self.traf
        del self.sim
        del self.keyb
        del self.cmd
        del self.scr
        del self.df
        pg.quit()
        return

#------------------------------------------------------------
# Main program: create tmx object, run it and delete it
#------------------------------------------------------------
tmx = TMX()      # Create object: initialisation
tmx.run()        # Run it
# del tmx         # Tidy up and exit, comment out for checking variables in shell

# Close metrics file explicitly for now
# needs to be moved to destructor traf.__del__  !

print "Closing Metrics file"
tmx.traf.metric.file.close()  

# Quit pygame
pg.quit()    # Only needed for debugging, normally part of del scr

print
print "BlueSky normal end."
