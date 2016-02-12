try:
    # Try Qt5 first
    from PyQt5.QtCore import QThread, QObject, QCoreApplication
except ImportError:
    # Else PyQt4 imports
    from PyQt4.QtCore import QThread, QObject, QCoreApplication
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

        # Flag indicating running at fixed rate or fast time
        self.ffmode      = False
        self.ffstop      = None

        # Simulation objects
        self.screenio    = ScreenIO(self)
        self.traf        = Traffic(navdb)
        self.stack       = Commandstack(self, self.traf, self.screenio)
        self.telnet_in   = StackTelnetServer(self.stack)
        self.navdb       = navdb
        # Metrics
        self.metric      = None
        # self.metric      = Metric()
        self.beastfeed     = Modesbeast(self.stack, self.traf)

    def moveToThread(self, target_thread):
        self.screenio.moveToThread(target_thread)
        self.telnet_in.moveToThread(target_thread)
        #self.beastfeed.moveToThread(target_thread)
        super(Simulation, self).moveToThread(target_thread)

    def doWork(self):
        # Start the telnet input server for stack commands
        self.telnet_in.start()

        self.syst = int(time.time() * 1000.0)
        self.fixdt = self.simdt

        while not self.mode == Simulation.end:
            # Timing bookkeeping
            self.samplecount += 1

            # Update the Mode-S beast parsing
            self.beastfeed.update()

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
            if not self.ffmode:
                self.syst += self.sysdt
                remainder = self.syst - int(1000.0 * time.time())

                if remainder > 0:
                    QThread.msleep(remainder)
            elif self.ffstop is not None and self.simt >= self.ffstop:
                self.start()

    def stop(self):
        self.mode = Simulation.end
        self.screenio.postQuit()

    def start(self):
        if self.ffmode:
            self.syst = int(time.time() * 1000.0)
        self.ffmode = False
        self.mode   = self.op

    def pause(self):
        self.mode   = self.hold

    def reset(self):
        self.simt   = 0.0
        self.mode   = self.init
        self.traf.reset(self.navdb)

    def fastforward(self, nsec=None):
        self.ffmode = True
        if nsec is not None:
            self.ffstop = self.simt + nsec
        else:
            self.ffstop = None

    def datafeed(self, flag):
        if flag == "ON":
            self.beastfeed.connectToHost(settings.modeS_host,
                                         settings.modeS_port)
        if flag == "OFF":
            self.beastfeed.disconnectFromHost()
