# -*- coding: utf-8 -*-
import numpy as np

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
        for v in self.StrVars:
            self.Vars[v].append('')
            
        for v in self.NumVars:
            self.Vars[v] = np.append(self.Vars[v],0)

    def DeleteElement(self,idx):
        for v in self.StrVars:
            del self.Vars[v][idx]
            
        for v in self.NumVars:
            self.Vars[v] = np.delete(self.Vars[v],idx)
