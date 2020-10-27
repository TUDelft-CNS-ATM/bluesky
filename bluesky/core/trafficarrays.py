""" Classes that derive from TrafficArrays (like Traffic) get automated create,
    delete, and reset functionality for all registered child arrays."""
# -*- coding: utf-8 -*-
try:
    from collections.abc import Collection
except ImportError:
    # In python <3.3 collections.abc doesn't exist
    from collections import Collection
import numpy as np

defaults = {"float": 0.0, "int": 0, "uint":0, "bool": False, "S": "", "str": ""}


class RegisterElementParameters:
    """ Class to use in 'with'-syntax. This class automatically
        calls for the _init_trafarrays function of the
        DynamicArray, with all parameters defined in 'with'."""

    def __init__(self, parent):
        self._parent = parent
        self.keys0 = set(parent.__dict__.keys())

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, tb):
        self._parent._init_trafarrays(set(self._parent.__dict__.keys()) - self.keys0)


class TrafficArrays:
    """ Parent class to use separate arrays and lists to allow
        vectorizing but still maintain and object like benefits
        for creation and deletion of an element for all parameters"""

    # The TrafficArrays class keeps track of all of the constructed
    # TrafficArray objects
    root = None
    ntraf = 0

    @staticmethod
    def setroot(obj):
        ''' This function is used to set the root of the tree of TrafficArray
            objects (which is the traffic object.)'''
        TrafficArrays.root = obj

    def __init__(self):
        super().__init__()
        self._parent   = TrafficArrays.root
        if self._parent:
            self._parent._children.append(self)
        self._children = []
        self._ArrVars  = []
        self._LstVars  = []

    def reparent(self, newparent):
        ''' Give TrafficArrays object a new parent. '''
        # Remove myself from the parent list of children, and add to new parent
        self._parent._children.pop(self._parent._children.index(self))
        newparent._children.append(self)
        self._parent = newparent

    def settrafarrays(self):
        ''' Convenience function for with-style traffic array registration. '''
        return RegisterElementParameters(self)

    def _init_trafarrays(self, keys):
        for key in keys:
            if isinstance(self.__dict__[key], list):
                self._LstVars.append(key)
            elif isinstance(self.__dict__[key], np.ndarray):
                self._ArrVars.append(key)
            elif isinstance(self.__dict__[key], TrafficArrays):
                self.__dict__[key].reparent(self)

        # In plugins and replaceable classes it could be that their instance
        # is created when the simulation is already running, and traffic is
        # present. Size traffic arrays accordingly here
        if TrafficArrays.root.ntraf:
            self.create(TrafficArrays.root.ntraf)

    def create(self, n=1):
        ''' Append n elements (aircraft) to all lists and arrays. '''

        for v in self._LstVars:  # Lists (mostly used for strings)
            lst = self.__dict__.get(v)
            vartype = type(lst[0]).__name__ if lst else 'str'
            lst.extend([defaults.get(vartype)] * n)

        for v in self._ArrVars:  # Numpy array
            # Get type without byte length
            vartype = ''.join(c for c in str(self.__dict__[v].dtype) if c.isalpha())
            self.__dict__[v] = np.append(self.__dict__[v], [defaults.get(vartype, 0)] * n)

    def istrafarray(self, name):
        ''' Returns true if parameter 'name' is a traffic array. '''
        return name in self._LstVars or name in self._ArrVars

    def create_children(self, n=1):
        ''' Call create (aircraft create) on all children. '''
        for child in self._children:
            child.create(n)
            child.create_children(n)

    def delete(self, idx):
        ''' Aircraft delete. '''
        # Remove element (aircraft) idx from all lists and arrays
        for child in self._children:
            child.delete(idx)

        for v in self._ArrVars:
            self.__dict__[v] = np.delete(self.__dict__[v], idx)

        if self._LstVars:
            if isinstance(idx, Collection):
                for i in reversed(idx):
                    for v in self._LstVars:
                        del self.__dict__[v][i]
            else:
                for v in self._LstVars:
                    del self.__dict__[v][idx]

    def reset(self):
        ''' Delete all elements from arrays and start at 0 aircraft. '''
        for child in self._children:
            child.reset()

        for v in self._ArrVars:
            self.__dict__[v] = np.array([], dtype=self.__dict__[v].dtype)

        for v in self._LstVars:
            self.__dict__[v] = []
