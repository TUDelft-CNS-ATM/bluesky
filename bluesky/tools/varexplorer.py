''' BlueSky variable explorer

    Provide flexible access to simulation data in BlueSky.
'''
from numbers import Number
from collections import OrderedDict
try:
    from collections.abc import Collection
except ImportError:
    # In python <3.3 collections.abc doesn't exist
    from collections import Collection
import re
import numpy as np
import bluesky as bs
from bluesky.tools import TrafficArrays

# Globals
# The variable lists and their corresponding sources
varlist = OrderedDict()


def init():
    ''' Variable explorer initialization function.
        Is called in bluesky.init() '''
    # Add the default sources to plot
    varlist.update([('sim', (bs.sim, getvarsfromobj(bs.sim))),
                    ('traf', (bs.traf, getvarsfromobj(bs.traf)))])


def register_data_parent(obj, name):
    varlist[name] = (obj, getvarsfromobj(obj))


def getvarsfromobj(obj):
    ''' Return a list with the numeric variables of the passed object.'''
    try:
        # Return attribute names, but exclude private attributes
        return [name for name in vars(obj) if not name[0] == '_']
    except TypeError:
        return None


def lsvar(varname=''):
    if not varname:
        return True, '\n' + \
        str.join(', ', [key for key in varlist])
    v = findvar(varname)
    print(v)
    if v:
        thevar = getattr(v.parent, v.varname)
        attrs = getvarsfromobj(thevar)
        vartype = thevar.__class__.__name__
        if isinstance(v.parent, TrafficArrays) and v.parent.istrafarray(v.varname):
            vartype += ' (TrafficArray)'
        txt = \
        'Variable:   {}\n'.format(v.varname) + \
        'Type:       {}\n'.format(vartype)
        if isinstance(thevar, Collection):
            txt += 'Size:       {}\n'.format(len(thevar))
        txt += 'Parent:     {}'.format(v.parentname)
        if attrs:
            txt += '\nAttributes: ' + str.join(', ', attrs) + '\n'
        return True, '\n' + txt
    return False, 'Variable {} not found'.format(varname)


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
                return Variable(obj, varset[-2][0], name, index)
        else:
            # A parent object is not passed, we only have a variable name
            # this name should exist in Plot.vlist
            for objname, objset in varlist.items():
                if name in objset[1]:
                    return Variable(objset[0], objname, name, index)
    except:
        pass
    return None


class Variable:
    def __init__(self, parent, parentname, varname, index):
        self.parent = parent
        self.parentname = parentname
        self.varname = varname
        try:
            self.index = [int(i) for i in index]
        except ValueError:
            self.index = []

    def is_num(self):
        ''' py3 replacement of operator.isNumberType.'''
        v = getattr(self.parent, self.varname)
        return isinstance(v, Number) or \
            (isinstance(v, np.ndarray) and v.dtype.kind not in 'OSUV')

    def get(self):
        if self.index:
            return getattr(self.parent, self.varname)[self.index]
        return getattr(self.parent, self.varname)
