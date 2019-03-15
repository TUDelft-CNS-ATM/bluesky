# Logic for group commands
import numpy as np
import bluesky as bs
from bluesky.tools.trafficarrays import TrafficArrays, RegisterElementParameters
from bluesky.tools import areafilter

class GroupArray(np.ndarray):
    # Similar to normal numpy arrays, but with the attribute of a groupname
    def __new__(cls,*args,groupname = '', **kwargs):
        ret = np.array(*args, **kwargs).view(cls)
        ret.groupname = groupname
        return ret
        
class TrafficGroups(TrafficArrays):
    def __init__(self):
        # Initialize the groups structure
        super(TrafficGroups, self).__init__()
        self.names = []
        self.ngroups = 0
        with RegisterElementParameters(self):
            self.ingroup = np.array([], dtype=np.uint32)

    def __contains__(self, groupname):
        # Check if a group with a name exists
        return groupname in self.names

    def Create(self, groupname, areaname = None):
        # Create a new group and possibly add members
        if self.ngroups >= 64:
            return False, 'Maximum number of 64 groups reached'
        if groupname in self.names:
            return False, 'Group %s alreay exists' %groupname

        # Add to list of names
        if '' in self.names:
            self.names[self.names.index('')] = groupname
        else:
            self.names.append(groupname)
        self.ngroups += 1

        # Add members already
        if areaname:
            if areafilter.hasArea(areaname):
                inside = areafilter.checkInside(areaname,bs.traf.lat,bs.traf.lon,bs.traf.alt)
                self.AddMembers(groupname,inside)
            else:
                return False, "Area %s does not exist" % areaname.upper()

    def deletegroup(self,grouparray):
        # Delete a group from memory, and kill all aircraft inside
        bs.traf.delete(grouparray)

        self.names[self.names.index(grouparray.groupname)] = ''
        self.ngroups -= 1

    def gmask(self,groupname):
        return (1<<self.names.index(groupname))

    def AddMembers(self,groupname,*args):
        # Add members to the group by aircraft id
        if groupname not in self:
            return False, "Group %s does not exist" %groupname
        self.ingroup[args] |= self.gmask(groupname)

    def RemoveMembers(self, groupname, *args):
        # Remove members from the group by aircraft id
        if groupname not in self:
            return False, "Group %s does not exist" %groupname
        self.ingroup[args] ^= self.gmask(groupname)

    def Membersidx(self, groupname):
        # Find aircraft id numbers of all members
        if groupname not in self:
            return False, "Group %s does not exist" %groupname
        return GroupArray(np.where(self.ingroup & self.gmask(groupname) > 0)[0],groupname=groupname)
        
    def ListMembers(self, groupname):
        # Used for echoing member names on screen
        if groupname not in self:
            return False, "Group %s does not exist" %groupname
        acnames = np.array(bs.traf.id)[self.Membersidx(groupname)]
        return True, ' '.join(acnames)

    def ListGroups(self):
        # Used for echoing all group names
        return True, ', '.join([n for n in self.names if n!=''])