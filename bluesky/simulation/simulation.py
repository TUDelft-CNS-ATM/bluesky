''' BlueSky simulation control object. '''
import time
import datetime
import signal
import numpy as np
from random import seed

# Local imports
import bluesky as bs
from bluesky.core.base import Base
import bluesky.core as core
from bluesky.core import plugin, simtime
from bluesky.network import subscriber
from bluesky.network.publisher import state_publisher, StatePublisher
from bluesky.core.walltime import Timer
from bluesky.core.timedfunction import hooks
from bluesky.stack import simstack, recorder
from bluesky.tools import datalog, areafilter, plotter


# Minimum sleep interval
MINSLEEP = 1e-3

# Register settings defaults
bs.settings.set_variable_defaults(simdt=0.05)

class Simulation(Base):
    ''' The simulation object. '''
    pub_simstate = StatePublisher('STATECHANGE')

    def __init__(self):
        super().__init__()
        self.state = bs.INIT
        self.prevstate = None

        # System time [seconds]
        self.syst = -1.0

        # Benchmark time and timespan [seconds]
        self.bencht = 0.0
        self.benchdt = -1.0

        # Simulation time [seconds]
        self.simt = 0.0

        # Simulation timestep [seconds]
        self.simdt = bs.settings.simdt

        # Simulation timestep multiplier: run sim at n x speed
        self.dtmult = 1.0

        # Simulated UTC clock time
        self.utc = datetime.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

        # Flag indicating running at fixed rate or fast time
        self.ffmode = False
        self.ffstop = None

        # Flag indicating whether timestep can be varied to ensure realtime op
        self.rtmode = False

        # Keep track of known clients
        self.clients = set()

        # Connect to system ABORT/INTERRUPT signal
        signal.signal(signal.SIGINT, lambda *args: self.quit())
        signal.signal(signal.SIGTERM, lambda *args: self.quit())

    @pub_simstate.payload
    def get_state(self):
        return dict(simstate=self.state)

    def run(self):
        ''' Start the main loop of this simulation. '''
        while self.state != bs.END:
            # Process timers
            Timer.update_timers()
            # Update network connections
            bs.net.update()
            # Perform a simulation step
            self.update()

            # Update screen logic
            bs.scr.update()

        # Close up properly on exit
        bs.net.close()
        datalog.reset()

        # Close savefile which may be open for recording
        recorder.saveclose()  # Close reording file if it is on

    def step(self, dt_increment=0):
        ''' Perform one simulation timestep.
        
            Call this function instead of update if you don't want to run with a fixed
            real-time rate.
        '''
        if self.state == bs.INIT:
            # Simulation starts as soon as there is traffic, or pending commands
            if bs.traf.ntraf > 0 or len(bs.stack.get_scendata()[0]) > 0:
                self.op()

        # Always update stack
        simstack.process()

        if self.state == bs.OP:
            # Plot/log the current timestep, and call preupdate functions
            plotter.update()
            datalog.update()
            hooks.preupdate.trigger()

            # Determine interval towards next timestep                
            self.simt, self.simdt = simtime.step(dt_increment)

            # Update UTC time
            self.utc += datetime.timedelta(seconds=self.simdt)

            # Update traffic and other update functions for the next timestep
            bs.traf.update()
            hooks.update.trigger()

        else:
            hooks.hold.trigger()

    def update(self):
        ''' Perform a simulation update. 
            This involves performing a simulation step, and when running in real-time mode
            (or a multiple thereof), sleeping an appropriate time. '''
        if self.state == bs.INIT:
            if self.syst < 0.0:
                self.syst = time.time()

            if self.benchdt > 0.0:
                self.fastforward(self.benchdt)
                self.bencht = time.time()

        # When running at a fixed rate, or when in hold/init,
        # increment system time with sysdt and calculate remainder to sleep.
        remainder = self.syst - time.time()
        if (not self.ffmode or self.state != bs.OP) and remainder > MINSLEEP:
            time.sleep(remainder)

        # Perform one simulation timestep
        if remainder < 0.0 and self.rtmode:
            # Allow a variable timestep when we are running realtime
            self.step(-remainder)
        else:
            # Don't accumulate delay when we aren't running realtime
            if remainder < 0:
                self.syst -= remainder
            self.step()

        # Always update syst
        self.syst += self.simdt / self.dtmult

        # Stop fast-time/benchmark if enabled and set interval has passed
        if self.ffstop is not None and self.simt >= self.ffstop:
            if self.benchdt > 0.0:
                simstack.echo('Benchmark complete: %d samples in %.3f seconds.' %
                            (bs.scr.samplecount, time.time() - self.bencht))
                self.benchdt = -1.0
                self.hold()
            else:
                self.op()

        # Inform main of our state change
        if self.state != self.prevstate:
            self.pub_simstate.send_replace(to_group=bs.net.server_id)
            self.prevstate = self.state

    def quit(self):
        ''' Quit simulation.
            This function is called when a QUIT signal is received from
            the server, or when quit is called. '''
        print(f'Simulation node {bs.net.node_id} quitting.')
        self.state = bs.END

    def op(self):
        ''' Set simulation state to OPERATE. '''
        self.syst = time.time() + self.simdt
        self.ffmode = False
        self.ffstop = None
        self.state = bs.OP
        self.set_dtmult(1.0)

    def hold(self):
        ''' Set simulation state to HOLD. '''
        self.syst = time.time() + self.simdt / self.dtmult
        self.state = bs.HOLD
        self.ffmode = False
        self.ffstop = None


    def reset(self):
        ''' Reset all simulation objects. '''
        self.state = bs.INIT
        self.syst = -1.0
        self.simt = 0.0
        self.simdt = bs.settings.simdt
        simtime.reset()
        self.utc = datetime.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        self.ffmode = False
        self.set_dtmult(1.0)
        hooks.reset.trigger()
        core.reset()
        bs.ref.reset()
        bs.navdb.reset()
        bs.traf.reset()
        simstack.reset()
        datalog.reset()
        areafilter.reset()
        bs.scr.reset()
        plotter.reset()

        # Communicate that this simulation has reset
        bs.net.send(b'RESET')

    def set_dtmult(self, mult):
        ''' Set simulation speed multiplier. '''
        self.dtmult = mult

    def realtime(self, flag=None):
        if flag is not None:
            self.rtmode = flag

        return True, 'Realtime mode is o' + ('n' if self.rtmode else 'ff')

    def fastforward(self, nsec=None):
        ''' Run in fast-time (for nsec seconds if specified). '''
        self.state = bs.OP
        self.ffmode = True
        self.ffstop = (self.simt + nsec) if nsec else None

    def benchmark(self, fname='IC', dt=300.0):
        ''' Run a simulation benchmark.
            Use scenario given by fname.
            Run for <dt> seconds. '''
        simstack.ic(fname)
        self.bencht  = 0.0  # Start time will be set at next sim cycle
        self.benchdt = dt

    def batch(self, fname):
        ''' Run a batch of scenarios. '''
        # The contents of the scenario file are meant as a batch list:
        # send to server and clear stack
        self.reset()
        try:
            scentime, scencmd = zip(*[tc for tc in simstack.readscn(fname)])
            bs.net.send(b'BATCH', (scentime, scencmd), bs.net.server_id)
        except FileNotFoundError:
            return False, f'BATCH: File not found: {fname}'

        return True

    @subscriber(topic='BATCH', broadcast=False)
    def start_batch_scenario(self, name, scentime, scencmd):
        ''' Start a scenario coming from the server batch. '''
        # We are in a batch simulation, and received an entire scenario. Assign it to the stack.
        self.reset()
        bs.stack.set_scendata(scentime, scencmd)
        self.op()

    @state_publisher(topic='SIMSETTINGS')
    def pub_simsettings(self):
        ''' Publish simulation settings on request. '''
        return dict(settings=bs.settings._settings_hierarchy,
            plugins=list(plugin.Plugin.plugins.keys()))

    def setutc(self, *args):
        ''' Set simulated clock time offset. '''
        if not args:
            pass  # avoid error message, just give time

        elif len(args) == 1:
            if args[0].upper() == 'RUN':
                self.utc = datetime.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

            elif args[0].upper() == 'REAL':
                self.utc = datetime.datetime.today().replace(microsecond=0)

            elif args[0].upper() == 'UTC':
                self.utc = datetime.datetime.utcnow().replace(microsecond=0)

            else:
                try:
                    self.utc = datetime.datetime.strptime(args[0], 
                        '%H:%M:%S.%f' if '.' in args[0] else '%H:%M:%S')
                except ValueError:
                    return False, 'Input time invalid'

        elif len(args) == 3:
            day, month, year = args
            try:
                self.utc = datetime.datetime(year, month, day)
            except ValueError:
                return False, 'Input date invalid.'
        elif len(args) == 4:
            day, month, year, timestring = args
            try:
                self.utc = datetime.datetime.strptime(
                    f'{year},{month},{day},{timestring}',
                    '%Y,%m,%d,%H:%M:%S.%f' if '.' in timestring else
                    '%Y,%m,%d,%H:%M:%S')
            except ValueError:
                return False, 'Input date invalid.'
        else:
            return False, 'Syntax error'

        return True, 'Simulation UTC ' + str(self.utc)

    @staticmethod
    def setseed(value):
        ''' Set random seed for this simulation. '''
        seed(value)
        np.random.seed(value)
