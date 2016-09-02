# -*- coding: utf-8 -*-
import numpy as np

defaults = {"float":0.0,"int":0,"bool":False,"S":"","str":""}

class DynamicArrays():
    """ Parent class to use separate arrays and lists to allow
        vectorizing but still maintain and object like benefits
        for creation and deletion of an element for all paramters"""

    def StartElementParameters(self):
        self.OriginalVars = []
        for var in self.__dict__:
            self.OriginalVars.append(var)

    def EndElementParameters(self):
        self.Vars = self.__dict__
        Nums=[]
        Strs=[]
        
        for var,val in self.Vars.iteritems():
            if var not in self.OriginalVars:
                if type(val)== list:
                    Strs.append(var)
                elif type(val) == np.ndarray:
                    Nums.append(var)
            
        # Only define self.NumVars & StrVars AFTER the loop, or else they are part of self.__dict__!
        self.NumVars=Nums
        self.StrVars=Strs

    def CreateElement(self):
        for v in self.StrVars:  # Lists (mostly used for strings)

            # Get type 
            if len(v)>0:
                vartype = str(type(v[0])).strip("<type '").strip("'>")

            if vartype in defaults:
                defaultvalue = defaults[vartype] 
            else:
                defaultvalue = ""
                
            self.Vars[v].append(defaultvalue)
            
        for v in self.NumVars: # Numpy array 

            # Get type without byte length
            fulltype = v.dtype
            vartype  = ""
            for c in fulltype:
                if not c.isdigit(): 
                    vartype = vartype + c
            
            # Get default value
            if vartype in defaults:
                defaultvalue = defaults[vartype]
            else:
                defaultvalue = 0.0
            
            self.Vars[v] = np.append(self.Vars[v],defaultvalue)

    def DeleteElement(self,idx):
        for v in self.StrVars:
            del self.Vars[v][idx]
            
        for v in self.NumVars:
            self.Vars[v] = np.delete(self.Vars[v],idx)
