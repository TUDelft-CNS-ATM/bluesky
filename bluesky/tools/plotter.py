''' Sim-side implementation of graphical data plotter in BlueSky.'''
from numbers import Number
from collections import OrderedDict
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
    varlist.update([('sim', getvarsfromobj(bs.sim)), ('traf', getvarsfromobj(bs.traf))])

def plot(*args):
    ''' Stack function to select a set of variables to plot.'''
    try:
        plots.append(Plot(*args))
        return True
    except IndexError as e:
        return False, e.args[0]

def update(simt):
    ''' Periodic update function for the plotter. '''
    pass

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
            obj = varlist.get(varset[0][0])
            for objname, _ in varset[1:-1]:
                if obj is None:
                    break
                obj = obj[objname] if obj in vars(obj) else None

            if obj and name in vars(obj):
                return obj, name, index
        else:
            # A parent object is not passed, we only have a variable name
            # this name should exist in Plot.vlist
            for el in varlist.values():
                if name in el[1]:
                    return el[0], name, index
    except:
        pass
    return None


class Plot(object):
    ''' A plot object.
        Each plot object is used to manage the plot of one variable
        on the sim side.'''
    def __init__(self, varx='', vary='', dt=1.0, color=None, fig=None):
        self.x = findvar(varx)
        self.y = findvar(vary)
        self.dt = dt
        self.color = color
        self.fig = fig

        if None in (self.x, self.y):
            raise IndexError('Variable %s not found' % (varx if self.x is None else vary))

        print('Created plot: x =', self.x, 'y =', self.y)
