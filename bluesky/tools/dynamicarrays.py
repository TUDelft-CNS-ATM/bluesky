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


class DynamicArrays(object):
    """ Parent class to use separate arrays and lists to allow
        vectorizing but still maintain and object like benefits
        for creation and deletion of an element for all paramters"""

    def MakeParameterLists(self, keys):
        self.Vars = self.__dict__
        ArrVars   = []
        Lsts      = []
        DynArrs   = []

        for key in keys:
            if type(self.Vars[key]) == list:
                Lsts.append(key)
            elif type(self.Vars[key]) == np.ndarray:
                ArrVars.append(key)
            elif isinstance(self.Vars[key], DynamicArrays):
                DynArrs.append(key)

        # Only define self.ArrVars & LstVars AFTER the loop, or else they are part of self.__dict__!
        self.ArrVars = ArrVars
        self.LstVars = Lsts
        self.DynArrs = DynArrs

    def create(self):
        # Append one element (aircraft) to all lists and arrays

        for v in self.LstVars:  # Lists (mostly used for strings)

            # Get type
            if len(v) > 0:
                vartype = str(type(v[0])).strip("<type '").strip("'>")

            if vartype in defaults:
                defaultvalue = defaults[vartype]
            else:
                defaultvalue = ""

            self.Vars[v].append(defaultvalue)

        for v in self.ArrVars:  # Numpy array
            # Get type without byte length
            fulltype = str(self.Vars[v].dtype)
            vartype = ""
            for c in fulltype:
                if not c.isdigit():
                    vartype = vartype + c

            # Get default value
            if vartype in defaults:
                defaultvalue = defaults[vartype]
            else:
                defaultvalue = 0.0

            self.Vars[v] = np.append(self.Vars[v], defaultvalue)

        for v in self.DynArrs:
            pass
            # The dynamic arrays refer to traf.parameter[-1] in their
            # .create functions, so first set all traf.parameter[-1]
            # to the right value and AFTER that perform create for all
            # dynamic arrays manually

    def delete(self, idx):
        # Remove element (aircraft) idx from all lists and arrays
        for v in self.LstVars:
            del self.Vars[v][idx]

        for v in self.ArrVars:
            self.Vars[v] = np.delete(self.Vars[v], idx)

        for v in self.DynArrs:
            self.Vars[v].delete(idx)

    def reset(self):
        # Delete all elements from arrays and start at 0 aircraft
        for v in self.LstVars:
            self.Vars[v] = []

        for v in self.ArrVars:
            self.Vars[v] = np.array([], dtype=self.Vars[v].dtype)

        for v in self.DynArrs:
            self.Vars[v].reset()
