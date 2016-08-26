""" BlueSky Datalogger """
import os
from operator import isNumberType as isnum
from datetime import datetime
import numpy as np
from .. import settings
from .. import stack

# Check if logdir exists, and if not, create it.
if not os.path.exists(settings.log_path):
    print 'Creating log path [' + settings.log_path + ']'
    os.makedirs(settings.log_path)

logprecision = '%.8f'

# Lists to contain the definitions of event and periodic logs
eventlogs    = dict()
periodiclogs = dict()


def createPeriodicLog(name, header, logdt, *variables):
    if name in periodiclogs:
        return False, name + ' already exists.'
    stackcmd = {name : [
        name + ' ON/OFF,[dt] or LISTVARS or SELECTVARS var1,...,varn',
        'txt,[float/txt,...]', lambda *args : setPeriodicLog(name, args)]
    }
    stack.append_commands(stackcmd)

    periodiclogs[name] = PeriodicLog(header, variables, logdt)


def createEventLog(name, header, variables):
    """ This function creates the event log dictionary. Call it in the __init__
    function of the class where the desired event logger is going to be used """
    # Eventlogs is a dict with the separate event logs, that each have a header,
    # a set of variables with their format, and a reference to the current file
    # (which starts as None)
    if name in eventlogs:
        return False

    # Add the access function for this logger to the stack command list
    stackcmd = {name : [name + ' ON/OFF', 'txt',
        lambda *args : setEventLog(name, args)]}
    stack.append_commands(stackcmd)

    # Create the log object in the eventlog dict
    varnames, formats = zip(*variables)
    header = header + '\n' + str.join(', ', varnames)
    eventlogs[name] = [None, header, tuple(formats)]


def logEvent(name, *data):
    """ This function is used to write to file the details of a desrired event
    when it occurs. It is to be called at the location where such event could occur"""
    log = eventlogs[name]
    formats = log[2]
    eventfile = log[0]
    if eventfile is None:
        return
    eventfile.write(str.join(', ', formats) % data)


def update(simt):
    """ This function writes to files of all periodic logs by calling the appropriate
    functions for each type of periodic log, at the approriate update time. """
    for key, log in periodiclogs.iteritems():
        log.log(simt)


def reset():
    """ This function closes all logs. It is called when simulation is
    reset and at quit. """

    # Close all periodic logs and remove reference to its file object
    for key, log in periodiclogs.iteritems():
        log.reset()

    # Close all event logs and remove reference to its file object
    for key, log in eventlogs.iteritems():
        if log[0]:
            log[0].close()
            log[0] = None


def makeLogfileName(logname):
    timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    scn = stack.get_scenfile()
    scn = scn[:scn.lower().find('.scn')]
    fname = "%s-[%s]-[%s].log" % (logname, scn, timestamp)
    return settings.log_path + '/' + fname


def setEventLog(logname, args):
    if logname not in eventlogs:
        return False, logname + " doesn't exist."

    log = eventlogs[logname]

    if len(args) == 0:
        return True, logname + ' is ' + ('OFF' if log[0] is None else 'ON')
    elif args[0] == 'ON':
        if log[0] is not None:
            log[0].close()

        log[0] = open(makeLogfileName(logname), 'w')
        log[0].write(log[1] + '\n')
    elif args[0] == 'OFF':
        if log[0] is not None:
            log[0].close()

    return True


def setPeriodicLog(logname, args):
    log = periodiclogs[logname]

    if len(args) == 0:
        return True, logname + ' is ' + ('ON' if log.isopen() else 'OFF')
    elif args[0] == 'ON':
        # Set log dt if passed
        if len(args) > 1:
            if type(args[1]) is float:
                log.dt = args[1]
            else:
                return False, 'Turn ' + logname + ' on with optional dt'

        log.open(makeLogfileName(logname))

    elif args[0] == 'OFF':
        log.reset()
    elif args[0] == 'LISTVARS':
        return True, 'Periodic log ' + logname + ' has variables: ' \
            + log.listallvarnames()
    elif args[0] == 'SELECTVARS':
        log.selectvars(args[1:])

    return True

# TODO: Eventlog also with class, create parent generic log class, try to make also log() common
# class EventLog:
#     def __init__(self, header, variables):
#         self.file       = None
#         self.header     = header
#         self.variables  = variables


class PeriodicLog:
    def __init__(self, header, variables, dt):
        self.file       = None
        self.header     = header
        self.allvars    = variables
        self.selvars    = variables
        self.dt         = dt
        self.default_dt = dt
        self.tlog       = 0.0

    def selectvars(self, selection):
        self.selvars = []
        selection = set(selection)
        for logset in self.allvars:
            cursel    = set(logset[1]) & selection
            if len(cursel) > 0:
                selection = selection - cursel
                self.selvars.append((logset[0], list(cursel)))

    def open(self, fname):
        if self.file:
            self.file.close()
        self.file       = open(fname, 'w')
        self.file.write(self.header + '\n')

    def isopen(self):
        return self.file is not None

    def log(self, simt):
        if self.file and len(self.selvars) > 0 and simt >= self.tlog:
            # Set the next log timestep
            self.tlog += self.dt

            # Make the variable reference list
            varlist = [logset[0].__dict__.get(varname) for logset in self.selvars for varname in logset[1]]

            # Convert numeric arrays to text, leave text arrays untouched
            txtdata = [len(varlist[0]) * ['%.3f' % simt]] + \
                [np.char.mod(logprecision, col) if isnum(col[0]) else col for col in varlist]

            # log the data to file
            np.savetxt(self.file, np.vstack(txtdata).T, delimiter=',', newline='\n', fmt='%s')

    def reset(self):
        self.dt         = self.default_dt
        self.tlog       = 0.0
        self.selvars    = self.allvars
        if self.file:
            self.file.close()
            self.file   = None

    def listallvarnames(self):
        ret = []
        for logset in self.allvars:
            ret.append(str.join(', ', logset[1]))
        return str.join(', ', ret)
