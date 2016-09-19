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

# Dict to contain the definitions of periodic loggers
periodicloggers = dict()

# Dict to contain all loggers (also the periodic loggers)
allloggers      = dict()


def registerLogParameters(name, dataparent):
    if name not in allloggers:
        allloggers[name] = CSVLogger(name)

    allloggers[name].dataparents.append(dataparent)
    return allloggers[name]


def defineLogger(name, header):
    if name not in allloggers:
        allloggers[name] = CSVLogger(name)

    allloggers[name].setheader(header)
    return allloggers[name]


def definePeriodicLogger(name, header, logdt):
    logger = defineLogger(name, header)
    logger.setdt(logdt)
    periodicloggers[name] = logger


def preupdate(simt):
    CSVLogger.simt = simt


def postupdate():
    """ This function writes to files of all periodic logs by calling the appropriate
    functions for each type of periodic log, at the approriate update time. """
    for key, log in periodicloggers.iteritems():
        log.log()


def reset():
    """ This function closes all logs. It is called when simulation is
    reset and at quit. """

    CSVLogger.simt = 0.0

    # Close all logs and remove reference to its file object
    for key, log in allloggers.iteritems():
        log.reset()


def makeLogfileName(logname):
    timestamp = datetime.now().strftime('%Y%m%d_%H-%M-%S')
    scn = stack.get_scenfile()
    scn = scn[:scn.lower().find('.scn')]
    fname = "%s_%s_%s.log" % (logname, scn, timestamp)
    return settings.log_path + '/' + fname


class CSVLogger:
    # Simulation time is static, shared between all loggers
    simt = 0.0

    def __init__(self, name):
        self.name        = name
        self.file        = None
        self.dataparents = []
        self.header      = ''
        self.tlog        = 0.0
        self.allvars     = []
        self.selvars     = []

        # In case this is a periodic logger: log timestep
        self.dt          = 0.0
        self.default_dt  = 0.0

        # Register a command for this logger in the stack
        stackcmd = {name : [
            name + ' ON/OFF,[dt] or LISTVARS or SELECTVARS var1,...,varn',
            '[txt,float/txt,...]', self.stackio]
        }
        stack.append_commands(stackcmd)

    def __enter__(self):
        self.keys0         = set(self.dataparents[-1].__dict__.keys())

    def __exit__(self, ex_type, ex_value, traceback):
        keys               = list(set(self.dataparents[-1].__dict__.keys()) - self.keys0)
        # Pre-process variable list to remove non-existing variables
        variables          = [x for x in keys if self.dataparents[-1].__dict__.get(x) is not None]
        self.allvars.append((self.dataparents[-1], variables))
        self.selvars.append((self.dataparents[-1], variables))
        self.dataparents.pop()

    def setheader(self, header):
        self.header     = header.split('\n')

    def setdt(self, dt):
        self.dt         = dt
        self.default_dt = dt

    def selectvars(self, selection):
        self.selvars = []
        for logset in self.allvars:
            # Create a list of member variables in logset that are in the selection
            cursel    = filter(lambda el: el.upper() in selection, logset[1])
            if len(cursel) > 0:
                # Add non-empty result with parent object to selected log variables
                self.selvars.append((logset[0], list(cursel)))

    def open(self, fname):
        if self.file:
            self.file.close()
        self.file       = open(fname, 'w')
        # Write the header
        for line in self.header:
            self.file.write('# ' + line + '\n')
        # Write the column contents
        columns = ['simt']
        for logset in self.selvars:
            columns += logset[1]
        self.file.write('# ' + str.join(', ', columns) + '\n')

    def isopen(self):
        return self.file is not None

    def log(self, *additional_vars):
        if self.file and len(self.selvars) > 0 and self.simt >= self.tlog:
            # Set the next log timestep
            self.tlog += self.dt

            # Make the variable reference list
            varlist = [v[0].__dict__.get(vname) for v in self.selvars for vname in v[1]] 
            varlist += additional_vars

            # Convert numeric arrays to text, leave text arrays untouched
            txtdata = [len(varlist[0]) * ['%.3f' % self.simt]] + \
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

    def stackio(self, *args):
        if len(args) == 0:
            text = 'This is '
            if self.name in periodicloggers:
                text += 'a periodic logger, with an update interval of %.2f seconds.\n' % self.dt
            else:
                text += 'a non-periodic logger.\n'
            text += self.name + ' is ' + ('ON' if self.isopen() else 'OFF') + \
                '\nUsage: ' + self.name + ' ON/OFF,[dt] or LISTVARS or SELECTVARS var1,...,varn'
            return True, text
        elif args[0] == 'ON':
            # Set log dt if passed
            if len(args) > 1:
                if type(args[1]) is float:
                    self.dt = args[1]
                else:
                    return False, 'Turn ' + self.name + ' on with optional dt'

            self.open(makeLogfileName(self.name))

        elif args[0] == 'OFF':
            self.reset()
        elif args[0] == 'LISTVARS':
            return True, 'Logger ' + self.name + ' has variables: ' \
                + self.listallvarnames()
        elif args[0] == 'SELECTVARS':
            self.selectvars(args[1:])

        return True
