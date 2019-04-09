import time, datetime
import bluesky as bs
from bluesky.tools import datalog, areafilter, plugin
from bluesky.tools.misc import txt2tim,tim2txt
from bluesky import stack
from bluesky.traffic.metric import Metric

onedayinsec = 24*3600 # [s] time of one day in seconds for clock time

class Simulation:
    """
    Simulation class definition : simulation control (time, mode etc.) class

    Methods:
        Simulation()            : constructor

        update()                : update sim variables (like time)
        op()                    : go from IC/HOLD to OPERATE mode
        pause()                 : go to HOLD mode (pause)
        stop()                  : quit function

    Created by  : Jacco M. Hoekstra (TU Delft)
    """

    # simulation modes
    init, hold, op, end = list(range(4))

    def __init__(self, detached):
        # simmode
        self.mode   = self.init

        self.simt   = 0.0   # Runtime
        self.tprev  = 0.0
        self.syst0  = 0.0
        self.simdt  = 0.0
        self.syst   = 0.0   # system time

        # Simulated UTC clock time
        self.utc = datetime.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

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


    def update(self):

        self.syst = time.clock()

        if self.mode == Simulation.init:
            self.operate()

        # Closk for run(=op) mode
        if self.mode == Simulation.op:

            # Not fast forward: variable dt
            if not self.ffmode:
                self.tprev = self.simt
                self.simt = self.syst - self.syst0
                self.simdt = self.simt - self.tprev

                # Protect against incidental dt's larger than 1 second,
                # due to window moving, switch to/from full screen, etc.
                if self.simdt > 1.0:
                    extra = self.simdt-1.0
                    self.simdt = 1.0
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

            # Update UTC time
            self.utc += datetime.timedelta(seconds=self.simdt)

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

    def addnodes(self, count):
        return

    def pause(self):  # Hold mode
        self.mode  = self.hold
        self.syst0 = self.syst-self.simt
        self.simdt = 0.0
        return

    def stop(self):  # Quit mode
        self.mode   = self.end
        datalog.reset()
        bs.stack.saveclose() # Save close configuration
#        datalog.save()
        return

    def operate(self):  # Back to op-mode: run after HOLD/PAUSE
        self.mode  = self.op
        self.syst  = time.clock()
        self.syst0 = self.syst-self.simt
        self.tprev = self.simt-0.001  # allow 1 msec step rto avoid div by zero
        return

    def setdt(self, dt):
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
        plugin.reset()
        bs.navdb.reset()
        bs.traf.reset()
        datalog.reset()
        areafilter.reset()
        self.utc = datetime.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    def setutc(self, *args):
        """ Set simulated clock time offset"""
        if not args:
            pass  # avoid error message, just give time

        elif len(args) == 1:
            if args[0].upper() == "RUN":
                self.utc = datetime.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

            elif args[0].upper() == "REAL":
                self.utc = datetime.datetime.today().replace(microsecond=0)

            elif args[0].upper() == "UTC":
                self.utc = datetime.datetime.utcnow().replace(microsecond=0)

            else:
                try:
                    self.utc = datetime.datetime.strptime(args[0], '%H:%M:%S.%f')
                except ValueError:
                    return False, "Input time invalid"

        elif len(args) == 3:
            day, month, year = args
            try:
                self.utc = datetime.datetime(year, month, day)
            except ValueError:
                return False, "Input date invalid."
        elif len(args) == 4:
            day, month, year, timestring = args
            try:
                self.utc = datetime.datetime.strptime(f'{year},{month},{day},{timestring}', '%Y,%m,%d,%H:%M:%S.%f')
            except ValueError:
                return False, "Input date invalid."
        else:
            return False, "Syntax error"

        return True, "Simulation UTC " + str(self.utc)
