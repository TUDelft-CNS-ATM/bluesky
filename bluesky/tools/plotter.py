''' Sim-side implementation of graphical data plotter in BlueSky.'''
from numbers import Number
from collections import OrderedDict, defaultdict
import re
import numpy as np
import bluesky as bs

# Globals
# The variable lists and their corresponding sources
varlist = OrderedDict()
# The list of plots
plots = list()


def init():
    ''' Plotter initialization function. Is called in bluesky.init() '''
    # Add the default sources to plot
    varlist.update([('sim', getvarsfromobj(bs.sim)),
                    ('traf', getvarsfromobj(bs.traf))])


def register_data_parent(obj, name):
    varlist[name] = getvarsfromobj(obj)


def plot(*args):
    ''' Stack function to select a set of variables to plot.
        Arguments: varx, vary, dt, color, fig. '''
    try:
        plots.append(Plot(*args))
        return True
    except IndexError as e:
        return False, e.args[0]


def update(simt):
    ''' Periodic update function for the plotter. '''
    streamdata = defaultdict(dict)
    for plot in plots:
        if plot.tnext <= simt:
            plot.tnext += plot.dt
            streamdata[plot.stream_id][plot.fig] = (plot.x.get(), plot.y.get(), plot.color)

    for streamname, data in streamdata.items():
        bs.net.send_stream(streamname, data)


def getvarsfromobj(obj):
    ''' Return a list with the numeric variables of the passed object.'''
    def is_num(o):
        ''' py3 replacement of operator.isNumberType.'''
        return isinstance(o, Number) or \
            (isinstance(o, np.ndarray) and o.dtype.kind not in 'OSUV')
    return (obj, [name for name in vars(obj) if is_num(vars(obj)[name])])


def findvar(varname):
    ''' Find a variable and its parent object in the registered varlist set, based
        on varname, as passed by the stack.
        Variables can be searched in two ways:
        By name only: e.g., varname lat returns (traf, lat)
        By object: e.g., varname traf.lat returns (traf, lat)
        '''
    try:
        # Find a string matching 'a.b.c[d]', where everything except a is optional
        varset = re.findall(r'(\w+)(?<=.)*(?:\[(\w+)\])?', varname.lower())
        # The actual variable is always the last
        name, index = varset[-1]
        # is a parent object passed? (e.g., traf.lat instead of just lat)
        if len(varset) > 1:
            # The first object should be in the varlist of Plot
            obj = varlist.get(varset[0][0])[0]
            # Iterate over objectname,index pairs in varset
            for pair in varset[1:-1]:
                if obj is None:
                    break
                obj = getattr(obj, pair[0], None)

            if obj and name in vars(obj):
                return Variable(obj, name, index)
        else:
            # A parent object is not passed, we only have a variable name
            # this name should exist in Plot.vlist
            for el in varlist.values():
                if name in el[1]:
                    return Variable(el[0], name, index)
    except:
        pass
    return None


class Variable:
    def __init__(self, parent, varname, index):
        self.parent = parent
        self.varname = varname
        try:
            self.index = [int(i) for i in index]
        except ValueError:
            self.index = []

    def get(self):
        if self.index:
            return getattr(self.parent, self.varname)[self.index]
        return getattr(self.parent, self.varname)


class Plot(object):
    ''' A plot object.
        Each plot object is used to manage the plot of one variable
        on the sim side.'''

    maxfig = 0

    def __init__(self, varx='', vary='', dt=1.0, color=None, fig=None):
        self.x = findvar(varx if vary else 'simt')
        self.y = findvar(vary or varx)
        self.dt = dt
        self.tnext = bs.sim.simt
        self.color = color
        if not fig:
            fig = Plot.maxfig
            Plot.maxfig += 1
        elif fig > Plot.maxfig:
            Plot.maxfig = fig

        self.fig = fig

        self.stream_id = b'PLOT' + bs.stack.sender()

        if None in (self.x, self.y):
            raise IndexError('Variable %s not found' % (varx if self.x is None else vary))
