try:
    # Try Qt5 first
    from PyQt5.QtCore import QThread, QObject
except ImportError:
    # Else PyQt4 imports
    from PyQt4.QtCore import QThread, QObject
import time

# Local imports
from screenio import ScreenIO
from simevents import StackTextEventType, BatchEventType, BatchEvent, SimStateEvent, SimQuitEventType
from ...traf import Traffic
from ...navdb import Navdatabase
from ...stack import Commandstack
from ...traf import Metric
from ... import settings
from ...tools.datafeed import Modesbeast
from ...tools.datalog import Datalog


class Simulation(QObject):
    # simulation modes
    init, op, hold, end = range(4)

    # =========================================================================
    # Functions
    # =========================================================================
    def __init__(self, manager):
        super(Simulation, self).__init__()
        self.manager     = manager
        self.running     = True
        self.state       = Simulation.init
        self.prevstate   = None

        # Set starting system time [milliseconds]
        self.syst        = 0.0

        # Benchmark time and timespan [seconds]
        self.bencht      = 0.0
        self.benchdt     = -1.0

        # Starting simulation time [seconds]
        self.simt        = 0.0

        # Simulation timestep [seconds]
        self.simdt       = settings.simdt

        # Simulation timestep multiplier: run sim at n x speed
        self.dtmult      = 1.0

        # System timestep [milliseconds]
        self.sysdt       = int(self.simdt / self.dtmult * 1000)

        # Flag indicating running at fixed rate or fast time
        self.ffmode      = False
        self.ffstop      = None

        # If available, name of the currently running scenario
        self.scenname    = 'Untitled'

        # Create datalog instance
        self.datalog = Datalog(self)

        # Simulation objects
        self.navdb       = Navdatabase('global')
        self.screenio    = ScreenIO(self, manager)
        self.traf        = Traffic(self.navdb)
        self.stack       = Commandstack(self, self.traf, self.screenio)

        # Additional modules
        self.metric      = Metric()
        self.beastfeed   = Modesbeast(self.stack, self.traf)

    def doWork(self):
        self.syst  = int(time.time() * 1000.0)
        self.fixdt = self.simdt

        while self.running:
            # Update screen logic
            self.screenio.update()

            # Update the Mode-S beast parsing
            self.beastfeed.update()

            # Simulation starts as soon as there is traffic, or pending commands
            if self.state == Simulation.init:
                if self.traf.ntraf > 0 or len(self.stack.scencmd) > 0:
                    self.start()
                    if self.benchdt > 0.0:
                        self.fastforward(self.benchdt)
                        self.bencht = time.time()

            if self.state == Simulation.op:
                self.stack.checkfile(self.simt)

            # Always update stack
            self.stack.process(self, self.traf, self.screenio)

            if self.state == Simulation.op:
                self.traf.update(self.simt, self.simdt)

                # Update metrics
                self.metric.update(self)

                # Update log
                self.datalog.update(self)

                # Update time for the next timestep
                self.simt += self.simdt

            # Process Qt events
            self.manager.processEvents()

            # When running at a fixed rate, or when in hold/init, increment system time with sysdt and calculate remainder to sleep
            if not self.ffmode or not self.state == Simulation.op:
                self.syst += self.sysdt
                remainder = self.syst - int(1000.0 * time.time())

                if remainder > 0:
                    QThread.msleep(remainder)
            elif self.ffstop is not None and self.simt >= self.ffstop:
                if self.benchdt > 0.0:
                    self.screenio.echo('Benchmark complete: %d samples in %.3f seconds.' % (self.screenio.samplecount, time.time() - self.bencht))
                    self.benchdt = -1.0
                    self.pause()
                else:
                    self.start()

            # Inform main of our state change
            if not self.state == self.prevstate:
                self.sendState()
                self.prevstate = self.state

    def stop(self):
        self.state   = Simulation.end
        self.datalog.save()

    def start(self):
        if self.ffmode:
            self.syst = int(time.time() * 1000.0)
        self.ffmode  = False
        self.state   = Simulation.op

    def pause(self):
        self.state   = Simulation.hold

    def reset(self):
        self.simt     = 0.0
        self.state    = Simulation.init
        self.ffmode   = False
        self.scenname = 'Untitled'
        self.traf.reset(self.navdb)
        self.stack.reset()
        self.screenio.reset()

    def quit(self):
        self.running = False

    def setDt(self, dt):
        self.simdt = abs(dt)
        self.sysdt = int(self.simdt / self.dtmult * 1000)

    def setDtMultiplier(self, mult):
        self.dtmult = mult
        self.sysdt = int(self.simdt / self.dtmult * 1000)

    def setFixdt(self, flag, nsec=None):
        if flag:
            self.fastforward(nsec)
        else:
            self.start()

    def fastforward(self, nsec=None):
        self.ffmode = True
        if nsec is not None:
            self.ffstop = self.simt + nsec
        else:
            self.ffstop = None

    def benchmark(self, fname='IC', dt=300.0):
        self.stack.ic(self.screenio, self, fname)
        self.bencht  = 0.0  # Start time will be set at next sim cycle
        self.benchdt = dt

    def scenarioInit(self, name):
        self.screenio.echo('Starting scenario ' + name)
        self.scenname = name

    def sendState(self):
        self.manager.sendEvent(SimStateEvent(self.state))

    def addNodes(self, count):
        self.manager.addNodes(count)

    def batch(self, filename):
        # The contents of the scenario file are meant as a batch list: send to manager and clear stack
        self.stack.openfile(filename)
        self.manager.sendEvent(BatchEvent(self.stack.scentime, self.stack.scencmd))
        self.reset()

    def event(self, event):
        # Keep track of event processing
        event_processed = False

        if event.type() == StackTextEventType:
            # We received a single stack command. Add it to the existing stack
            self.stack.stack(event.cmdtext)
            event_processed = True

        elif event.type() == BatchEventType:
            # We are in a batch simulation, and received an entire scenario. Assign it to the stack.
            self.reset()
            self.stack.scentime = event.scentime
            self.stack.scencmd  = event.scencmd
            self.start()
            event_processed     = True
        elif event.type() == SimQuitEventType:
            # BlueSky is quitting
            self.quit()
        else:
            # This is either an unknown event or a gui event.
            event_processed = self.screenio.event(event)

        return event_processed
