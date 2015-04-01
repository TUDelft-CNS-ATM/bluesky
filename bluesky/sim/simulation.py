import time
from ..tools.datalog import Datalog

class Simulation:
    """ 
    Simulation class definition : simulation control (time, mode etc.) class

    Methods:
        Simulation()            : constructor

        update()                : update sim variables (like time)
        start()                 : go from IC/HOLD to OPERATE mode
        pause()                 : go to HOLD mode (pause)
        stop()                  : quit function
       
    Created by  : Jacco M. Hoekstra (TU Delft)
    """
    
    init = 0
    op   = 1
    hold = 2
    end  = 3

    def __init__(self,tmx):

        # Access to other objects
        self.tmx = tmx

        # simmode
        self.mode   = self.init

        self.t      = 0.0   # Runtime 
        self.tprev  = 0.0
        self.syst0  = 0.0
        self.dt     = 0.0
        self.tsys   = 0.0 # system time

        # Directories
        self.datadir = "./data/"
        self.dts = []

        # Create datalog instance
        self.datalog = Datalog()

        # Fixed dt mode for fast forward
        self.ffmode = False  # Default FF off
        self.fixdt = 0.1     # Default time step
        self.ffstop = -1.    # Indefinitely
        return

    def update(self):

        self.tsys = time.clock()

        # Closk for run(=op) mode
        if self.mode==self.op:

            # Not fast forward: variable dt            
            if not self.ffmode:
                self.tprev = self.t
                self.t     = self.tsys - self.syst0
                self.dt = self.t - self.tprev

                # Protect against incidental dt's larger than 1 second,
                # due to window moving, switch to/from full screen, etc.
                if self.dt>1.0:
                    extra = self.dt-1.0
                    self.t = self.t - extra
                    self.syst0 = self.tsys-self.t
 
            # Fast forward: fixed dt until ffstop time, goto pause
            else:
                self.dt = self.fixdt
                self.t = self.t+self.fixdt
                self.syst0 = self.tsys - self.t
                if self.ffstop > 0. and self.t>=self.ffstop:
                    self.ffmode = False
                    self.mode = self.hold

            # For measuring game loop frequency                 
            self.dts.append(self.dt)
            if len(self.dts)>20:
                    del self.dts[0]

        # HOLD/Pause mode
        else:
            self.syst0 = self.tsys-self.t
            self.dt = 0.0

        return          

    def start(self):
        """Initial Condition mode, start running with t=0"""
        self.mode   = self.op
        self.tsys   = time.clock()
        self.syst0  = time.clock()
        self.t      = 0.0
        self.tprev  = 0.0
        return
    
    def pause(self):  # Hold mode
        self.mode = self.hold
        self.syst0 = self.tsys-self.t
        self.dt = 0.0
        return

    def stop(self):  # Quit mode
        self.mode   = self.end
        self.datalog.save()
        return
        
    def play(self): # Back to op-mode: run after HOLD/PAUSE
        self.mode = self.op
        self.tsys = time.clock()
        self.syst0 = self.tsys-self.t
        self.tprev = self.t-0.001 # allow 1 msec step rto avoid div by zero
        return
 