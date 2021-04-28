""" Conditional commands:
KL204 ATSPD 250 KL204 LNAV ON
KL204 ATALT FL100 KL204 SPD 350
"""
import numpy as np
import bluesky as bs
from bluesky import stack
from bluesky.tools.geo import qdrdist

# Enumerated condtion types
alttype, spdtype, postype = 0, 1, 2


class Condition():
    def __init__(self):

        self.ncond = 0  # Number of conditions

        self.id       = []                         # Id of aircraft of condition
        self.condtype = np.array([],dtype=int)     # Condition type (0=alt,1=spd)
        self.target   = np.array([],dtype=float)   # Target value (alt,speed,distance[nm])
        self.lastdif  = np.array([],dtype=float)   # Difference during last update
        self.posdata  = []                         # Data for postype: tuples lat[deg],lon[deg] of ref position
        self.cmd      = []                         # Commands to be issued

    def update(self):
        if self.ncond==0:
            return

        # Update indices based on list of id's
        acidxlst = np.array(bs.traf.id2idx(self.id))
        if len(acidxlst)>0:
            idelcond = sorted(list(np.where(acidxlst<0)[0]))
            for i in idelcond[::-1]:
                del (self.id[i])
                self.condtype = np.delete(self.condtype, i)
                self.target = np.delete(self.target, i)
                self.lastdif = np.delete(self.lastdif, i)
                del self.posdata[i]
                del self.cmd[i]

            self.ncond = len(self.id)
            if self.ncond==0:
                return
            acidxlst = np.array(bs.traf.id2idx(self.id))

        # Check condition types
        actdist = np.ones(self.ncond)*999e9  # Invalid number which never triggers anything is extremely large
        for j in range(self.ncond):
            if self.condtype[j] == postype:
                qdr,dist = qdrdist(bs.traf.lat[acidxlst[j]],bs.traf.lon[acidxlst[j]],self.posdata[j][0],self.posdata[j][1])
                actdist[j] = dist # [nm]

        # Get relevant actual value using index list as index to numpy arrays
        self.actual = (self.condtype == alttype) * bs.traf.alt[acidxlst] + \
                      (self.condtype == spdtype) * bs.traf.cas[acidxlst] + \
                      (self.condtype == postype) * actdist

        # Compare sign of actual difference with sign of last difference
        actdif       = self.target - self.actual

        # Make sorted arrya of indices of true conditions and their conditional commands
        idxtrue      = sorted(list(np.where(actdif*self.lastdif <= 0.0)[0]))# Sign changed
        self.lastdif = actdif
        if idxtrue==None or len(idxtrue)==0:
            return


        # Execute commands found to have true condition
        for i in idxtrue:
            if i>=0:
                stack.stack(self.cmd[i])
                # debug
                # stack.stack(" ECHO Conditional command issued: "+self.cmd[i])

        # Delete executed commands to clean up arrays and lists
        # from highest index to lowest for consistency
        for i in idxtrue[::-1]:
            if i>=0:
                del self.id[i]
                self.condtype = np.delete(self.condtype,i)
                self.target   = np.delete(self.target,i)
                self.lastdif  = np.delete(self.lastdif,i)
                del self.posdata[i]
                del self.cmd[i]

        # Adjust number of conditions
        self.ncond = len(self.id)

        if self.ncond!=len(self.cmd):
            print ("self.ncond=",self.ncond)
            print ("self.cmd=",self.cmd)
            print ("traffic/conditional.py: self.delcondition: invalid condition array size")
        return

    def ataltcmd(self,acidx,targalt,cmdtxt):
        actalt = bs.traf.alt[acidx]
        self.addcondition(acidx, alttype, targalt, actalt, cmdtxt)
        return True

    def atspdcmd(self, acidx, targspd, cmdtxt):
        actspd = bs.traf.tas[acidx]
        self.addcondition(acidx, spdtype, targspd, actspd,cmdtxt)
        return True

    def atdistcmd(self, acidx, lat, lon, targdist, cmdtxt):
        qdr, actdist = qdrdist(bs.traf.lat[acidx], bs.traf.lon[acidx], lat, lon)
        self.addcondition(acidx, postype, targdist, actdist, cmdtxt, (lat,lon))
        return True

    def addcondition(self,acidx, icondtype, target, actual, cmdtxt,latlon=None):
        #print ("addcondition:", acidx, icondtype, target, actual, cmdtxt, latlon)

        # Add condition to arrays
        self.id.append(bs.traf.id[acidx])

        self.condtype = np.append(self.condtype,icondtype)
        self.target   = np.append(self.target,target)
        self.lastdif  = np.append(self.lastdif,target - actual)

        self.posdata.append(latlon)
        self.cmd.append(cmdtxt)

        self.ncond = self.ncond+1
        #print("addcondition: self.ncond",self.ncond)
        return

    def renameac(self,oldid,newid):
        # Continonal commands are stored per id (ac name)
        # When renamed, call this method to update list
        # rename ids in list of ids
        # Call this if RENAME command is implemented
        if self.id.count(old.id) == 0:
            return
        for i in range(len(self.id)):
            if self.id[i] == oldid:
                self.id[i] = newid
        return





