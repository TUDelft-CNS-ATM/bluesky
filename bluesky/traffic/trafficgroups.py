# Logic for group commands
import numpy as np
import bluesky as bs
from bluesky.core import TrafficArrays
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
        super().__init__()
        self.groups = dict()
        self.allmasks = 0
        with self.settrafarrays():
            self.ingroup = np.array([], dtype=np.int64)

    def __contains__(self, groupname):
        # Check if a group with a name exists
        return groupname in self.groups or groupname == '*'

    def group(self, groupname='', *args):
        '''Add aircraft to group, list aircraft in group, or list existing groups.'''
        # Return list of groups if no groupname is given
        if not groupname:
            if not self.groups:
                return True, 'There are currently no traffic groups defined.'
            else:
                return True, 'Defined traffic groups:\n' + ', '.join(self.groups)
        if len(self.groups) >= 64:
            return False, 'Maximum number of 64 groups reached'
        if groupname not in self.groups:
            if not args:
                return False, f'Group {groupname} doesn\'t exist'
            # Get first unused group mask
            for i in range(64):
                groupmask = (1 << i)
                if not self.allmasks & groupmask:
                    self.allmasks |= groupmask
                    self.groups[groupname] = groupmask
                    break

        elif not args:
            acnames = np.array(bs.traf.id)[self.listgroup(groupname)]
            return True, 'Aircraft in group {}:\n{}'.format(groupname, ', '.join(acnames))

        # Add aircraft to group
        if areafilter.hasArea(args[0]):
            inside = areafilter.checkInside(
                args[0], bs.traf.lat, bs.traf.lon, bs.traf.alt)
            self.ingroup[inside] |= self.groups[groupname]
            acnames = np.array(bs.traf.id)[inside]
        else:
            idx = list(args)
            self.ingroup[idx] |= self.groups[groupname]
            acnames = np.array(bs.traf.id)[idx]
        return True, 'Aircraft added to group {}:\n{}'.format(groupname, ', '.join(acnames))

    def delgroup(self, grouparray):
        ''' Delete a group, and all aircraft in that group. '''
        # Delete all aircraft in the respective group
        bs.traf.delete(grouparray)

        # Remove the group from the group list
        if grouparray.groupname != '*':
            self.allmasks ^= self.groups.pop(grouparray.groupname)

    def ungroup(self, groupname, *args):
        ''' Remove members from the group by aircraft id. '''
        groupmask = self.groups.get(groupname, None)
        if groupmask is None:
            return False, f"Group {groupname} doesn't exist"
        self.ingroup[list(args)] ^= groupmask

    def listgroup(self, groupname):
        ''' Return aircraft index for all aircraft in group. 
            When * is passed as groupname, all aircraft in simulation are returned. '''
        if groupname == '*':
            return GroupArray(range(bs.traf.ntraf), groupname='*')
        groupmask = self.groups.get(groupname, None)
        if groupmask is None:
            return False, f"Group {groupname} doesn't exist"
        return GroupArray(np.where((self.ingroup & groupmask) > 0)[0], groupname=groupname)
