''' Remotestore provides automatic storage of data that needs to be individually
    updated and accessed for each individual remote node, and easy access of data of the active node.
'''
from types import SimpleNamespace
from numbers import Number
from copy import deepcopy
from collections import defaultdict

import bluesky as bs


# Keep track of default attribute values of mutable type.
# These always need to be stored per remote node.
defaults = SimpleNamespace()

# Keep a dict of remote state storage namespaces
remotes = defaultdict(lambda: deepcopy(defaults))


def get(remote_id=None, group=None):
    ''' Retrieve a remote store, or a group in a remote store.
        Returns the store of the active remote if no remote id is provided.
    '''
    return (remotes[remote_id or bs.net.act_id] if group is None else
            getattr(remotes[remote_id or bs.net.act_id], group))


def setvalue(name, value, remote_id=None, group=None):
    ''' Set the value of attribute 'name' in group 'group' for remote store with id 'remote_id' 
        Sets value in store of the active remote if no remote_id is provided.
    '''
    setattr(remotes[remote_id or bs.net.act_id] if group is None else
            getattr(remotes[remote_id or bs.net.act_id], group), name, value)


def setdefault(name, default, group=None):
    ''' Set the default value for variable 'name' in group 'group' '''
    setattr(defaults if group is None else getattr(defaults, group), name, default)


def addgroup(name):
    ''' Add a storage group to each remote data store. '''
    # Add store to the defaults
    setattr(defaults, name, SimpleNamespace())

    # Also add to existing stores if necessary
    for remote in remotes.values():
        setattr(remote, name, SimpleNamespace())


class ActData:
    ''' Access data from the active remote as if it is a member variable. '''
    __slots__ = ('default', 'name', 'group')

    def __init__(self, default, group=None):
        self.default = default
        self.name = ''
        self.group = group

    def __set_name__(self, owner, name):
        self.name = name
        # If underlying datatype is mutable, always immediately
        # store per remote node
        if not isinstance(self.default, (str, tuple, Number, frozenset, bytes)):
            setdefault(name, self.default, self.group)

            # In case remote data already exists, update stores
            for remote_id in remotes.keys():
                setvalue(name, deepcopy(self.default), remote_id, self.group)

    def __get__(self, obj, objtype=None):
        ''' Return the actual value for the currently active node. '''
        if not bs.net.act_id:
            return self.default
        return getattr(
            remotes[bs.net.act_id] if self.group is None else getattr(remotes[bs.net.act_id], self.group),
            self.name, self.default)
        # TODO: What is the active (client) node on the sim-side? Is this always within a currently processed stack command? -> stack.sender_id

    def __set__(self, obj, value):
        if not bs.net.act_id:
            self.default = value
        else:
            setattr(remotes[bs.net.act_id] if self.group is None else getattr(remotes[bs.net.act_id], self.group), self.name, value)
