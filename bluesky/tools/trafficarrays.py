""" Classes that derive from TrafficArrays (like Traffic) get automated create,
    delete, and reset functionality for all registered child arrays."""
# -*- coding: utf-8 -*-
import numpy as np

defaults = {"float": 0.0, "int": 0, "bool": False, "S": "", "str": ""}


class RegisterElementParameters():
    """ Class to use in 'with'-syntax. This class automatically
        calls for the MakeParameterLists function of the
        DynamicArray, with all parameters defined in 'with'."""

    def __init__(self, parent):
        self.parent = parent

    def __enter__(self):
        self.keys0 = set(self.parent.__dict__.keys())

    def __exit__(self, type, value, tb):
        self.parent.MakeParameterLists(set(self.parent.__dict__.keys()) - self.keys0)


class TrafficArrays(object):
    """ Parent class to use separate arrays and lists to allow
        vectorizing but still maintain and object like benefits
        for creation and deletion of an element for all paramters"""

    # The TrafficArrays class keeps track of all of the constructed
    # TrafficArray objects
    root = None

    @classmethod
    def SetRoot(cls, obj):
        ''' This function is used to set the root of the tree of TrafficArray
            objects (which is the traffic object.)'''
        cls.root = obj

    def __init__(self):
        self.parent   = TrafficArrays.root
        if self.parent:
            self.parent.children.append(self)
        self.children = []
        self.ArrVars  = []
        self.LstVars  = []
        self.Vars     = self.__dict__

    def reparent(self, newparent):
        # Remove myself from the parent list of children, and add to new parent
        self.parent.children.pop(self.parent.children.index(self))
        newparent.children.append(self)
        self.parent = newparent

    def MakeParameterLists(self, keys):
        for key in keys:
            if isinstance(self.Vars[key], list):
                self.LstVars.append(key)
            elif isinstance(self.Vars[key], np.ndarray):
                self.ArrVars.append(key)
            elif isinstance(self.Vars[key], TrafficArrays):
                self.Vars[key].reparent(self)

    def create(self, n=1):
        # Append one element (aircraft) to all lists and arrays

        for v in self.LstVars:  # Lists (mostly used for strings)

            # Get type
            vartype = None
            lst = self.__dict__.get(v)
            if len(lst) > 0:
                vartype = str(type(lst[0])).split("'")[1]

            if vartype in defaults:
                defaultvalue = [defaults[vartype]] * n
            else:
                defaultvalue = [""] * n

            self.Vars[v].extend(defaultvalue)

        for v in self.ArrVars:  # Numpy array
            # Get type without byte length
            fulltype = str(self.Vars[v].dtype)
            vartype = ""
            for c in fulltype:
                if not c.isdigit():
                    vartype = vartype + c

            # Get default value
            if vartype in defaults:
                defaultvalue = [defaults[vartype]] * n
            else:
                defaultvalue = [0.0] * n

            self.Vars[v] = np.append(self.Vars[v], defaultvalue)

    def create_children(self, n=1):
        for child in self.children:
            child.create(n)
            child.create_children(n)

    def delete(self, idx):
        # Remove element (aircraft) idx from all lists and arrays
        for v in self.LstVars:
            del self.Vars[v][idx]

        for v in self.ArrVars:
            self.Vars[v] = np.delete(self.Vars[v], idx)

        for child in self.children:
            child.delete(idx)

    def reset(self):
        # Delete all elements from arrays and start at 0 aircraft
        for v in self.LstVars:
            self.Vars[v] = []

        for v in self.ArrVars:
            self.Vars[v] = np.array([], dtype=self.Vars[v].dtype)

        for child in self.children:
            child.reset()
