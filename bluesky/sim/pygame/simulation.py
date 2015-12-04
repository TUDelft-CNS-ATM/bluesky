import time
from ...tools.datalog import Datalog
from ...traf import Traffic
from ...stack import Commandstack
# from ...traf import Metric

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

    # simulation modes
    init, op, hold, end = range(4)

    def __init__(self, gui, navdb):
        # simmode
        self.mode   = self.init

        self.simt   = 0.0   # Runtime
        self.tprev  = 0.0
        self.syst0  = 0.0
        self.dt     = 0.0
        self.syst   = 0.0   # system time

        # Directories
        self.datadir = "./data/"
        self.dts = []

        # Create datalog instance
        self.datalog = Datalog()

        # Fixed dt mode for fast forward
        self.ffmode = False  # Default FF off
        self.fixdt = 0.1     # Default time step
        self.ffstop = -1.    # Indefinitely

        # Simulation objects
        self.traf  = Traffic(navdb)
        self.navdb = navdb
        self.metric = None
        # self.metric = Metric()

        # Stack ties it together
        self.stack = Commandstack(self, self.traf, gui.scr)

        return

    def update(self, scr):

        self.syst = time.clock()

        if self.mode == Simulation.init:
            self.start()

        # Closk for run(=op) mode
        if self.mode == Simulation.op:

            # Not fast forward: variable dt
            if not self.ffmode:
                self.tprev = self.simt
                self.simt     = self.syst - self.syst0
                self.dt = self.simt - self.tprev

                # Protect against incidental dt's larger than 1 second,
                # due to window moving, switch to/from full screen, etc.
                if self.dt>1.0:
                    extra = self.dt-1.0
                    self.simt = self.simt - extra
                    self.syst0 = self.syst-self.simt
 
            # Fast forward: fixed dt until ffstop time, goto pause
            else:
                self.dt = self.fixdt
                self.simt = self.simt+self.fixdt
                self.syst0 = self.syst - self.simt
                if self.ffstop > 0. and self.simt>=self.ffstop:
                    self.ffmode = False
                    self.mode = self.hold

            # For measuring game loop frequency                 
            self.dts.append(self.dt)
            if len(self.dts)>20:
                    del self.dts[0]

            self.stack.checkfile(self.simt)

        # Always process stack
        self.stack.process(self, self.traf, scr)

        if self.mode == Simulation.op:
            self.traf.update(self.simt, self.dt)

            # Update metrics
            if self.metric is not None:
                self.metric.update(self, self.traf)

        # HOLD/Pause mode
        else:
            self.syst0 = self.syst-self.simt
            self.dt = 0.0

        return

    def pause(self):  # Hold mode
        self.mode = self.hold
        self.syst0 = self.syst-self.simt
        self.dt = 0.0
        return

    def stop(self):  # Quit mode
        self.mode   = self.end
        self.datalog.save()
        return

    def start(self):  # Back to op-mode: run after HOLD/PAUSE
        self.mode = self.op
        self.syst = time.clock()
        self.syst0 = self.syst-self.simt
        self.tprev = self.simt-0.001  # allow 1 msec step rto avoid div by zero
        return
    
    def reset(self):
        self.simt = 0.0
        self.mode = Simulation.init
        self.traf.reset(self.navdb)
