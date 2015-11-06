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
from ... import settings


class Simulation(QObject):
    # =========================================================================
    # Settings
    # =========================================================================
    # Simulation timestep [seconds]
    simdt = settings.simdt

    # Simulation loop update rate [Hz]
    sys_rate = settings.sim_update_rate

    # Flag indicating running at fixed rate or fast time
    run_fixed = True

    # simulation modes
    init, op, hold, end = range(4)

    # =========================================================================
    # Functions
    # =========================================================================
    def __init__(self, gui, navdb):
        super(Simulation, self).__init__()
        self.mode = Simulation.init

        self.samplecount = 0
        self.sysdt = 1000 / self.sys_rate

        # Simulation objects
        self.screenio = ScreenIO(self)
        self.traf = Traffic(navdb)
        self.navdb = navdb

        # Stack ties it all together
        self.stack = Commandstack(self,self.traf,gui)

        print 'Initializing multi-threaded simulation'

    def moveToThread(self, target_thread):
        self.screenio.moveToThread(target_thread)
        super(Simulation, self).moveToThread(target_thread)

    def doWork(self):
        # Set starting system time [milliseconds]
        self.syst = int(time.time() * 1000.0)

        # Set starting simulation time [seconds]
        self.simt  = 0.0

        while not self.mode == Simulation.end:
            # Timing bookkeeping
            self.samplecount += 1

            # TODO: what to do with init
            if self.mode == Simulation.init:
                self.mode = Simulation.op

            if self.mode == Simulation.op:
                self.stack.checkfile(self.simt)

            # Always update stack
            self.stack.process(self, self.traf, self.screenio)

            if self.mode == Simulation.op:
                self.traf.update(self.simt, self.simdt)

                # Update time for the next timestep
                self.simt += self.simdt

            # Process Qt events
            QCoreApplication.processEvents()

            # When running at a fixed rate, increment system time with sysdt and calculate remainder to sleep
            if self.run_fixed:
                self.syst += self.sysdt
                remainder = self.syst - int(1000.0 * time.time())

                if remainder > 0:
                    QThread.msleep(remainder)

    def stop(self):
        self.mode = Simulation.end
        # TODO: Communicate quit signal to main thread

    def start(self):
        self.mode = Simulation.op

    def pause(self):
        self.mode = Simulation.hold

    def reset(self):
        self.simt = 0.0
        self.mode = Simulation.init
        self.traf.reset(self.navdb)
