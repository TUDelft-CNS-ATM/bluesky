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
    # Add the default sources to the variable explorer
    varlist.update([('sim', (bs.sim, getvarsfromobj(bs.sim))),
                    ('traf', (bs.traf, getvarsfromobj(bs.traf)))])


def register_data_parent(obj, name):
    varlist[name] = (obj, getvarsfromobj(obj))


def getvarsfromobj(obj):
    ''' Return a list with the names of the variables of the passed object.'''
    try:
        # Return attribute names, but exclude private attributes
        return [name for name in vars(obj) if not name[0] == '_']
    except TypeError:
        return None


def lsvar(varname=''):
    ''' Stack function to list information on simulation variables in the
        BlueSky console. '''
    if not varname:
        # When no argument is passed, show a list of parent objects for which
        # variables can be accessed
        return True, '\n' + \
        str.join(', ', [key for key in varlist])

    # Find the variable in the variable list
    v = findvar(varname)
    if v:
        thevar = v.get()  # reference to the actual variable
        attrs = getvarsfromobj(thevar)  # When the variable is an object, get child attributes
        vartype = v.get_type()  # Type of the variable
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
            obj = None
            # The first object should be in the varlist of Plot
            # As either a top-level object:
            if varset[0][0] in varlist:
                obj = varlist.get(varset[0][0])[0]
            else:
                for objname, objset in varlist.items():
                    if varset[0][0] in objset[1]:
                        obj = getattr(objset[0], varset[0][0])

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
    ''' Wrapper class for variable explorer.
        Keeps reference to parent object, parent name, and variable name. '''
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
            (isinstance(v, np.ndarray) and v.dtype.kind not in 'OSUV') or \
            (isinstance(v, Collection) and self.index and
            all([isinstance(v[i], Number) for i in self.index]))

    def get_type(self):
        ''' Return the a string containing the type name of this variable. '''
        return self.get().__class__.__name__

    def get(self):
        ''' Get a reference to the actual variable. '''
        if self.index:
            v = getattr(self.parent, self.varname)
            return [v[i] for i in self.index]
        return getattr(self.parent, self.varname)
