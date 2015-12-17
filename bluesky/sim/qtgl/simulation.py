try:
    # Try Qt4 first
    from PyQt4.QtCore import QThread, QObject, QCoreApplication
except ImportError:
    # Else PyQt5 imports
    from PyQt5.QtCore import QThread, QObject, QCoreApplication
import time

# Local imports
from screenio import ScreenIO
from ...traf import Traffic
from ...stack import Commandstack
from ...tools.network import StackTelnetServer
# from ...traf import Metric
from ... import settings
from ...tools.datafeed import Modesbeast


class Simulation(QObject):
    # =========================================================================
    # Settings
    # =========================================================================
    # Simulation timestep [seconds]
    simdt = settings.simdt

    # Simulation loop update rate [Hz]
    sys_rate = settings.sim_update_rate

    # Flag indicating running at fixed rate or fast time
    run_fast = False

    # simulation modes
    init, op, hold, end = range(4)

    # =========================================================================
    # Functions
    # =========================================================================
    def __init__(self, gui, navdb):
        super(Simulation, self).__init__()
        print 'Initializing multi-threaded simulation'

        self.mode        = Simulation.init
        self.samplecount = 0
        self.sysdt       = 1000 / self.sys_rate

        # Set starting system time [milliseconds]
        self.syst        = 0.0

        # Starting simulation time [seconds]
        self.simt        = 0.0

        self.ff_end      = None

        # Simulation objects
        self.screenio    = ScreenIO(self)
        self.traf        = Traffic(navdb)
        self.stack       = Commandstack(self, self.traf, self.screenio)
        self.telnet_in   = StackTelnetServer(self.stack)
        #self.modes_in    = Modesbeast(self)
        self.navdb       = navdb
        # Metrics
        self.metric      = None
        # self.metric      = Metric()

    def moveToThread(self, target_thread):
        self.screenio.moveToThread(target_thread)
        self.telnet_in.moveToThread(target_thread)
        #self.modes_in.moveToThread(target_thread)
        super(Simulation, self).moveToThread(target_thread)

    def doWork(self):
        # Start the telnet input server for stack commands
        self.telnet_in.start()

        self.syst = int(time.time() * 1000.0)

        while not self.mode == Simulation.end:
            # Timing bookkeeping
            self.samplecount += 1

            # Update the Mode-S beast parsing
            #self.modes_in.update()

            # TODO: what to do with init
            if self.mode == Simulation.init:
                self.mode = Simulation.op

            if self.mode == Simulation.op:
                self.stack.checkfile(self.simt)

            # Always update stack
            self.stack.process(self, self.traf, self.screenio)

            if self.mode == Simulation.op:
                self.traf.update(self.simt, self.simdt)

                # Update metrics
                if self.metric is not None:
                    self.metric.update(self, self.traf)

                # Update time for the next timestep
                self.simt += self.simdt

            # Process Qt events
            QCoreApplication.processEvents()

            # When running at a fixed rate, increment system time with sysdt and calculate remainder to sleep
            if not self.run_fast:
                self.syst += self.sysdt
                remainder = self.syst - int(1000.0 * time.time())

                if remainder > 0:
                    QThread.msleep(remainder)
            elif self.ff_end is not None and self.simt >= self.ff_end:
                self.start()

    def stop(self):
        self.mode = Simulation.end
        # TODO: Communicate quit signal to main thread

    def start(self):
        if self.run_fast:
            self.syst = int(time.time() * 1000.0)
        self.run_fast = False
        self.mode     = Simulation.op

    def pause(self):
        self.mode     = Simulation.hold

    def reset(self):
        self.simt     = 0.0
        self.mode     = Simulation.init
        self.traf.reset(self.navdb)

    def fastforward(self, nsec=[]):
        self.run_fast = True
        if len(nsec) > 0:
            self.ff_end = self.simt + nsec[0]
        else:
            self.ff_end = None
