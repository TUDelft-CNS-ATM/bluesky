import time, datetime

# Local imports
import bluesky as bs
from bluesky import settings, stack
from bluesky.tools import datalog, areafilter, plugin, plotter, simtime
from bluesky.tools.misc import txt2tim, tim2txt


# Minimum sleep interval
MINSLEEP = 1e-3

onedayinsec = 24 * 3600  # [s] time of one day in seconds for clock time

# Register settings defaults
settings.set_variable_defaults(simdt=0.05, simevent_port=10000, simstream_port=10001)


def Simulation(detached):
    """ Return Simulation object either in normal network-attached mode
        or detached mode.
    """
    if detached:
        from bluesky.network.detached import Node
    else:
        from bluesky.network.node import Node

    class SimulationClass(Node):
        ''' The simulation object. '''
        def __init__(self):
            super(SimulationClass, self).__init__(settings.simevent_port,
                                                  settings.simstream_port)
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

            # Simulated UTC clock time
            self.utc = datetime.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

            # System timestep [seconds]
            self.sysdt = self.simdt / self.dtmult

            # Flag indicating running at fixed rate or fast time
            self.ffmode = False
            self.ffstop = None

        def step(self):
            ''' Perform a simulation timestep. '''
            super().step()
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

                # Update sim and UTC time for the next timestep
                self.simt, self.simdt = simtime.step()
                self.utc += datetime.timedelta(seconds=self.simdt)

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
            simtime.reset()
            self.utc = datetime.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            self.ffmode = False
            self.setDtMultiplier(1.0)
            plugin.reset()
            bs.navdb.reset()
            bs.traf.reset()
            stack.reset()
            datalog.reset()
            areafilter.reset()
            bs.scr.reset()

        def setdt(self, dt=None, target='simdt'):
            if dt is not None and target == 'simdt':
                self.simdt = abs(dt)
                self.sysdt = self.simdt / self.dtmult
            return simtime.setdt(dt, target)

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
                event_processed = True
            elif eventname == b'QUIT':
                # BlueSky is quitting
                self.stop()
            elif eventname == b'GETSIMSTATE':
                # Send list of stack functions available in this sim to gui at start
                stackdict = {cmd : val[0][len(cmd) + 1:] for cmd, val in stack.cmddict.items()}
                shapes = [shape.raw for shape in areafilter.areas.values()]
                simstate = dict(pan=bs.scr.def_pan, zoom=bs.scr.def_zoom,
                    stackcmds=stackdict, stacksyn=stack.cmdsynon, shapes=shapes,
                    custacclr=bs.scr.custacclr, custgrclr=bs.scr.custgrclr)
                self.send_event(b'SIMSTATE', simstate, target=sender_rte)
            else:
                # This is either an unknown event or a gui event.
                event_processed = bs.scr.event(eventname, eventdata, sender_rte)

            return event_processed

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
                        self.utc = datetime.datetime.strptime(args[0], 
                            '%H:%M:%S.%f' if '.' in args[0] else '%H:%M:%S')
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
                    self.utc = datetime.datetime.strptime(
                        f'{year},{month},{day},{timestring}',
                        '%Y,%m,%d,%H:%M:%S.%f' if '.' in timestring else
                        '%Y,%m,%d,%H:%M:%S')
                except ValueError:
                    return False, "Input date invalid."
            else:
                return False, "Syntax error"

            return True, "Simulation UTC " + str(self.utc)

    return SimulationClass()
