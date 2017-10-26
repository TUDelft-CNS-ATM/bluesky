import time

# Local imports
import bluesky as bs
from bluesky import settings, stack
# from bluesky.traffic import Metric
from bluesky.tools import datalog, areafilter, plugin
from bluesky.tools.misc import txt2tim, tim2txt
from bluesky.io import Node
from .simevents import StackTextEventType, BatchEventType, BatchEvent, \
    SimStateEvent, SimQuitEventType, StackInitEvent

# Minimum sleep interval
MINSLEEP = 2e-3

onedayinsec = 24 * 3600  # [s] time of one day in seconds for clock time

# Register settings defaults
settings.set_variable_defaults(simdt=0.05)

class Simulation(Node):
    ''' The simulation object. '''
    def __init__(self):
        super(Simulation, self).__init__()
        self.state       = bs.INIT
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

        # Simulated clock time
        self.deltclock   = 0.0
        self.simtclock   = self.simt

        # System timestep [milliseconds]
        self.sysdt       = int(self.simdt / self.dtmult * 1000)

        # Flag indicating running at fixed rate or fast time
        self.ffmode      = False
        self.ffstop      = None

        # Additional modules
        # self.metric      = Metric()

    def prepare(self):
        # Send list of stack functions available in this sim to gui at start
        stackdict = {cmd : val[0][len(cmd) + 1:] for cmd, val in stack.cmddict.items()}
        self.send_event(StackInitEvent(stackdict))
        self.syst = int(time.time() * 1000.0)

    def step(self):
        ''' Perform a simulation timestep. '''
        # When running at a fixed rate, or when in hold/init,
        # increment system time with sysdt and calculate remainder to sleep.
        if not self.ffmode or not self.state == bs.OP:
            # TODO: python sleep is floating point seconds
            remainder = self.syst / 1000.0 - time.time()
            if remainder > MINSLEEP:
                time.sleep(remainder)

        elif self.ffstop is not None and self.simt >= self.ffstop:
            if self.benchdt > 0.0:
                bs.scr.echo('Benchmark complete: %d samples in %.3f seconds.' % \
                            (bs.scr.samplecount, time.time() - self.bencht))
                self.benchdt = -1.0
                self.pause()
            else:
                self.op()

        if self.state == bs.OP:
            # Plugins pre-update
            plugin.preupdate(self.simt)

        # Update screen logic
        bs.scr.update()

        # Simulation starts as soon as there is traffic, or pending commands
        if self.state == bs.INIT:
            if bs.traf.ntraf > 0 or len(stack.get_scendata()[0]) > 0:
                self.op()
                if self.benchdt > 0.0:
                    self.fastforward(self.benchdt)
                    self.bencht = time.time()

        if self.state == bs.OP:
            stack.checkfile(self.simt)

        # Always update stack
        stack.process()

        if self.state == bs.OP:

            bs.traf.update(self.simt, self.simdt)

            # Update plugins
            plugin.update(self.simt)

            # Update loggers
            datalog.postupdate()

            # Update time for the next timestep
            self.simt += self.simdt

            # Update clock
            self.simtclock = (self.deltclock + self.simt) % onedayinsec

        # Always update syst
        self.syst += self.sysdt

        # Inform main of our state change
        if not self.state == self.prevstate:
            self.sendState()
            self.prevstate = self.state

    def stop(self):
        self.state = bs.END
        datalog.reset()

    def op(self):
        self.syst   = int(time.time() * 1000.0)
        self.ffmode = False
        self.state  = bs.OP

    def pause(self):
        self.state = bs.HOLD

    def reset(self):
        self.simt      = 0.0
        self.deltclock = 0.0
        self.simtclock = self.simt
        self.state     = bs.INIT
        self.ffmode    = False
        plugin.reset()
        bs.navdb.reset()
        bs.traf.reset()
        stack.reset()
        datalog.reset()
        areafilter.reset()
        bs.scr.reset()

    def setDt(self, dt):
        self.simdt = abs(dt)
        self.sysdt = int(self.simdt / self.dtmult * 1000)

    def setDtMultiplier(self, mult):
        self.dtmult = mult
        self.sysdt  = int(self.simdt / self.dtmult * 1000)

    def setFixdt(self, flag, nsec=None):
        if flag:
            self.fastforward(nsec)
        else:
            self.op()

    def fastforward(self, nsec=None):
        self.ffmode = True
        if nsec is not None:
            self.ffstop = self.simt + nsec
        else:
            self.ffstop = None

    def benchmark(self, fname='IC', dt=300.0):
        stack.ic(fname)
        self.bencht  = 0.0  # Start time will be set at next sim cycle
        self.benchdt = dt

    def sendState(self):
        self.send_event(SimStateEvent(self.state))

    def addNodes(self, count):
        # TODO Addnodes function
        pass
        # self.addNodes(count)

    def batch(self, filename):
        # The contents of the scenario file are meant as a batch list: send to manager and clear stack
        result = stack.openfile(filename)
        if result:
            scentime, scencmd = stack.get_scendata()
            self.send_event(BatchEvent(scentime, scencmd))
            self.reset()
        return result

    def event(self, event, sender_id):
        # Keep track of event processing
        event_processed = False

        if event.type() == StackTextEventType:
            # We received a single stack command. Add it to the existing stack
            stack.stack(event.cmdtext, sender_id)
            event_processed = True

        elif event.type() == BatchEventType:
            # We are in a batch simulation, and received an entire scenario. Assign it to the stack.
            self.reset()
            stack.set_scendata(event.scentime, event.scencmd)
            self.op()
            event_processed     = True
        elif event.type() == SimQuitEventType:
            # BlueSky is quitting
            self.quit()
        else:
            # This is either an unknown event or a gui event.
            event_processed = bs.scr.event(event, sender_id)

        return event_processed

    def setclock(self, txt=""):
        """ Set simulated clock time offset"""
        if txt == "":
            pass  # avoid error message, just give time

        elif txt.upper() == "RUN":
            self.deltclock = 0.0
            self.simtclock = self.simt

        elif txt.upper() == "REAL":
            tclock = time.localtime()
            self.simtclock = tclock.tm_hour * 3600. + tclock.tm_min * 60. + tclock.tm_sec
            self.deltclock = self.simtclock - self.simt

        elif txt.upper() == "UTC":
            utclock = time.gmtime()
            self.simtclock = utclock.tm_hour * 3600. + utclock.tm_min * 60. + utclock.tm_sec
            self.deltclock = self.simtclock - self.simt

        elif txt.replace(":", "").replace(".", "").isdigit():
            self.simtclock = txt2tim(txt)
            self.deltclock = self.simtclock - self.simt
        else:
            return False, "Time syntax error"

        return True, "Time is now " + tim2txt(self.simtclock)
