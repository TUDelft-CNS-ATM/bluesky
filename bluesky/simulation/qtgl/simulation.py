import time

# Local imports
import bluesky as bs
from bluesky import settings, stack
from bluesky.tools import datalog, areafilter, plugin, plotter
from bluesky.tools.misc import txt2tim, tim2txt
if settings.is_detached:
    from bluesky.network.detached import Node
else:
    from bluesky.network.node import Node

# Minimum sleep interval
MINSLEEP = 1e-3

onedayinsec = 24 * 3600  # [s] time of one day in seconds for clock time

# Register settings defaults
settings.set_variable_defaults(simdt=0.05)

class Simulation(Node):
    ''' The simulation object. '''
    def __init__(self):
        super(Simulation, self).__init__()
        self.state = bs.INIT
        self.prevstate = None

        # Set starting system time [seconds]
        self.syst = -1.0

        # Benchmark time and timespan [seconds]
        self.bencht = 0.0
        self.benchdt = -1.0

        # Starting simulation time [seconds]
        self.simt = 0.0

        # Simulation timestep [seconds]
        self.simdt = settings.simdt

        # Simulation timestep multiplier: run sim at n x speed
        self.dtmult = 1.0

        # Simulated clock time
        self.deltclock = 0.0
        self.simtclock = self.simt

        # System timestep [seconds]
        self.sysdt = self.simdt / self.dtmult

        # Flag indicating running at fixed rate or fast time
        self.ffmode = False
        self.ffstop = None

    def step(self):
        ''' Perform a simulation timestep. '''
        # When running at a fixed rate, or when in hold/init,
        # increment system time with sysdt and calculate remainder to sleep.
        if not self.ffmode or not self.state == bs.OP:
            remainder = self.syst - time.time()
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
            if self.syst < 0.0:
                self.syst = time.time()

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

            # Update Plotter
            plotter.update(self.simt)

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

        # Close savefile which may be open for recording
        bs.stack.saveclose()  # Close reording file if it is on

        self.quit()

    def op(self):
        self.syst = time.time()
        self.ffmode = False
        self.state = bs.OP
        self.setDtMultiplier(1.0)

    def pause(self):
        self.syst = time.time()
        self.state = bs.HOLD

    def reset(self):
        self.state = bs.INIT
        self.syst = -1.0
        self.simt = 0.0
        self.simdt = settings.simdt
        self.deltclock = 0.0
        self.simtclock = self.simt
        self.ffmode = False
        self.setDtMultiplier(1.0)
        plugin.reset()
        bs.navdb.reset()
        bs.traf.reset()
        stack.reset()
        datalog.reset()
        areafilter.reset()
        bs.scr.reset()

    def setDt(self, dt):
        self.simdt = abs(dt)
        self.sysdt = self.simdt / self.dtmult

    def setDtMultiplier(self, mult):
        self.dtmult = mult
        self.sysdt  = self.simdt / self.dtmult

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
        self.send_event(b'STATECHANGE', self.state)

    def batch(self, filename):
        # The contents of the scenario file are meant as a batch list: send to server and clear stack
        result = stack.openfile(filename)
        if result:
            scentime, scencmd = stack.get_scendata()
            self.send_event(b'BATCH', (scentime, scencmd))
            self.reset()
        return result

    def event(self, eventname, eventdata, sender_rte):
        # Keep track of event processing
        event_processed = False

        if eventname == b'STACKCMD':
            # We received a single stack command. Add it to the existing stack
            stack.stack(eventdata, sender_rte)
            event_processed = True

        elif eventname == b'BATCH':
            # We are in a batch simulation, and received an entire scenario. Assign it to the stack.
            self.reset()
            stack.set_scendata(eventdata['scentime'], eventdata['scencmd'])
            self.op()
            event_processed     = True
        elif eventname == b'QUIT':
            # BlueSky is quitting
            self.quit()
        elif eventname == b'GETSIMSTATE':
            # Send list of stack functions available in this sim to gui at start
            stackdict = {cmd : val[0][len(cmd) + 1:] for cmd, val in stack.cmddict.items()}
            shapes = [shape.raw for shape in areafilter.areas.values()]
            simstate = dict(pan=bs.scr.def_pan, zoom=bs.scr.def_zoom, stackcmds=stackdict, shapes=shapes)
            self.send_event(b'SIMSTATE', simstate, target=sender_rte)
        else:
            # This is either an unknown event or a gui event.
            event_processed = bs.scr.event(eventname, eventdata, sender_rte)

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
