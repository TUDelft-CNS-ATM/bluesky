import time
import bluesky as bs
from bluesky.tools import datalog, areafilter, plugin
from bluesky.tools.misc import txt2tim,tim2txt
from bluesky import stack
from bluesky.traffic.metric import Metric
from bluesky.tools.network import StackTelnetServer

onedayinsec = 24*3600 # [s] time of one day in seconds for clock time

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
    init, op, hold, end = list(range(4))

    def __init__(self):
        # simmode
        self.mode   = self.init

        self.simt   = 0.0   # Runtime
        self.tprev  = 0.0
        self.syst0  = 0.0
        self.simdt  = 0.0
        self.syst   = 0.0   # system time

        self.deltclock = 0.0   # SImulated clock time at simt=0.
        self.simtclock = 0.0

        # Directories
        self.datadir = "./data/"
        self.dts = []

        # Fixed dt mode for fast forward
        self.ffmode = False  # Default FF off
        self.fixdt = 0.1     # Default time step
        self.ffstop = -1.    # Indefinitely

        # Simulation objects
        print("Setting up Traffic simulation")
        self.metric = Metric()

        # Additional modules
        self.telnet_in   = StackTelnetServer()

    def update(self):

        self.syst = time.clock()

        if self.mode == Simulation.init:
            self.start()

        # Closk for run(=op) mode
        if self.mode == Simulation.op:

            # Not fast forward: variable dt
            if not self.ffmode:
                self.tprev = self.simt
                self.simt  = self.syst - self.syst0
                self.simdt = self.simt - self.tprev

                # Protect against incidental dt's larger than 1 second,
                # due to window moving, switch to/from full screen, etc.
                if self.simdt > 1.0:
                    extra = self.simdt-1.0
                    self.simt = self.simt - extra
                    self.syst0 = self.syst-self.simt

            # Fast forward: fixed dt until ffstop time, goto pause
            else:
                self.simdt = self.fixdt
                self.simt = self.simt+self.fixdt
                self.syst0 = self.syst - self.simt
                if self.ffstop > 0. and self.simt >= self.ffstop:
                    self.ffmode = False
                    self.mode = self.hold

            # Update simulated clock time
            self.simtclock = (self.simt + self.deltclock)%onedayinsec

            # Datalog pre-update (communicate current sim time to loggers)
            datalog.preupdate(self.simt)

            # Plugins pre-update
            plugin.preupdate(self.simt)

            # For measuring game loop frequency
            self.dts.append(self.simdt)
            if len(self.dts) > 20:
                    del self.dts[0]

            stack.checkfile(self.simt)

        # Always process stack
        stack.process()

        if self.mode == Simulation.op:
            bs.traf.update(self.simt, self.simdt)

            # Update metrics
            self.metric.update()

            # Update plugins
            plugin.update(self.simt)

            # Update loggers
            datalog.postupdate()

        # HOLD/Pause mode
        else:
            self.syst0 = self.syst-self.simt
            self.simdt = 0.0

        return

    def scenarioInit(self, name):
        self.reset()
        return

    def benchmark(self, fname='IC', tend=60.0):
        return False, "Benchmark command not available in Pygame version."

    def batch(self, filename):
        return False, "Batch comand not available in Pygame version," + \
                 "use Qt-version for batch simulations"

    def addNodes(self, count):
        return

    def pause(self):  # Hold mode
        self.mode  = self.hold
        self.syst0 = self.syst-self.simt
        self.simdt = 0.0
        return

    def stop(self):  # Quit mode
        self.mode   = self.end
        datalog.reset()
#        datalog.save()
        return

    def start(self):  # Back to op-mode: run after HOLD/PAUSE
        self.mode  = self.op
        self.syst  = time.clock()
        self.syst0 = self.syst-self.simt
        self.tprev = self.simt-0.001  # allow 1 msec step rto avoid div by zero
        return

    def setDt(self, dt):
        self.fixdt = abs(dt)

    def setDtMultiplier(self, mult=None):
        return False, "Dt multiplier not available in Pygame version."

    def setFixdt(self, flag=None, nsec=None):
        if flag is not None:
            if flag:
                self.fastforward(nsec)
            else:
                self.ffmode = False

    def fastforward(self, nsec=None):
        self.ffmode = True
        if nsec is not None:
            self.ffstop = self.simt + nsec
        else:
            self.ff_end = -1.0

    def reset(self):
        self.simt = 0.0
        self.mode = self.init
        bs.navdb.reset()
        bs.traf.reset()
        datalog.reset()
        areafilter.reset()
        self.delclock  = 0.0   # SImulated clock time at simt=0.
        self.simtclock = 0.0

    def setclock(self,txt=""):
        """ Set simulated clock time offset"""
        if txt == "":
            pass # avoid error message, just give time

        elif txt.upper()== "RUN":
            self.deltclock = 0.0
            self.simtclock = self.simt

        elif txt.upper()== "REAL":
            tclock = time.localtime()
            self.simtclock = tclock.tm_hour*3600. + tclock.tm_min*60. + tclock.tm_sec
            self.deltclock = self.simtclock - self.simt

        elif txt.upper()== "UTC":
            utclock = time.gmtime()
            self.simtclock = utclock.tm_hour*3600. + utclock.tm_min*60. + utclock.tm_sec
            self.deltclock = self.simtclock - self.simt

        elif txt.replace(":","").replace(".","").isdigit():
            self.simtclock = txt2tim(txt)
            self.deltclock = self.simtclock - self.simt
        else:
            return False,"Time syntax error"


        return True,"Time is now "+tim2txt(self.simtclock)
