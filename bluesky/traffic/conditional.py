""" Conditional commands:
KL204 ATSPD 250 KL204 LNAV ON
KL204 ATALT FL100 KL204 SPD 350
"""
import numpy as np
import bluesky as bs
from bluesky import stack

# Enumerated condtion types
alttype, spdtype = 0, 1


class Condition():
    def __init__(self):

        self.ncond = 0  # Number of conditions

        self.id       = np.chararray((0,0))         # Id of aicrcraft of condition
        self.idx      = np.array([],dtype=np.int)# Index of aicrcraft of condition
        self.condtype = []                       # Condition type (0=alt,1=spd)
        self.target   = np.array([])             # Target value
        self.lastdif  = np.array([])             # Difference during last update
        self.cmd      = []                       # Commands to be issued

    def update(self):
        if self.ncond==0:
            return

        # Check a/c index
        # checkcorrel() # Not necessary of self.delac is used

        # Check condition types
        # Get relevant actual value

        self.actual = (self.condtype == alttype) * bs.traf.alt[self.idx] + \
                      (self.condtype == spdtype) * bs.traf.cas[self.idx]

        # Compare sign of actual difference with sign of last difference
        actdif       = self.target - self.actual
        idxtrue      = np.where(actdif*self.lastdif <= 0.0)[0] # Sign changed
        self.lastdif = actdif
        if len(idxtrue)==0:
            return

        # Execute commands found to have true condition
        for i in idxtrue:
            stack.stack(self.cmd[i])

        self.delcondition(idxtrue) # Remove when executed

    def ataltcmd(self,acidx,targalt,cmdtxt):
        actalt = bs.traf.alt[acidx]
        self.addcondition(acidx, alttype, targalt, actalt, cmdtxt)
        return True

    def atspdcmd(self, acidx, targspd, cmdtxt):
        actspd = bs.traf.tas[acidx]
        self.addcondition(acidx,spdtype,targspd,actspd,cmdtxt)
        return True

    def addcondition(self,acidx, icondtype, target, actual, cmdtxt):
        #print ("addcondition:", acidx, icondtype, target, actual, cmdtxt)

        # Add condition to arrays
        self.id       = np.append(self.id,bs.traf.id[acidx])
        self.idx      = np.append(self.idx,acidx)
        self.condtype = np.append(self.condtype,icondtype)
        self.target   = np.append(self.target,target)
        self.lastdif  = np.append(self.lastdif,target - actual)
        self.cmd.append(cmdtxt)

        self.ncond = self.ncond+1
        #print("addcondition: self.ncond",self.ncond)

    def delcondition(self,idelarray): # Delete conditions with indices in this list
        if self.ncond==0:
            return

        #print("delcondition: idelarray=", idelarray)
        #print("self.ncond=", self.ncond)
        #print("self.id =", self.id)
        #print("self.cmd=", self.cmd)

        self.id       = np.delete(self.id, idelarray)
        self.idx      = np.delete(self.idx, idelarray)
        self.condtype = np.delete(self.condtype, idelarray)
        self.target   = np.delete(self.target, idelarray)
        self.lastdif  = np.delete(self.lastdif, idelarray)

        # Update command list
        newcmd = []
        for i in range(len(self.cmd)):
            if i not in idelarray:
                newcmd.append(self.cmd[i])
        self.cmd = newcmd

        # Adjust number of conditions
        self.ncond = len(self.id)

        if self.ncond!=len(self.cmd):
            print ("self.ncond=",self.ncond)
            print ("self.cmd=",self.cmd)
            print ("traffic/conditional.py: self.delcondition: invalid condition array size")

    #    def checkcorrel(self):
#        wrongidx = np.where(self.id!=np.chararray(bs.traf.id))

    def delac(self, acidx):
        if self.ncond==0:
            return

        # Check one or more?
        if type(acidx)==int:
            self.deloneac(acidx)
        else:
            for idx in acidx:
                self.deloneac(idx)

    def deloneac(self,acidx): # Delete one aircraft from conditoon database

        # Take care of deleted aircraft condition
        idel = np.where(self.idx==acidx)[0]

        if len(idel)>0:
            self.delcondition(idel)

        # Update indices above
        self.idx = np.where(self.idx <= acidx, self.idx, self.idx - 1)

