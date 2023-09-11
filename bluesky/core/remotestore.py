''' Remotestore provides automatic storage of data that needs to be individually
    updated and accessed for each individual remote node, and easy access of data of the active node.
'''
from types import SimpleNamespace
from numbers import Number
from copy import deepcopy
from collections import defaultdict
from functools import partial

import bluesky as bs


def _genstore():
    store = deepcopy(defaults)
    for g in generators:
        g(store)
    return store


class Store(SimpleNamespace):
    def valid(self):
        ''' Return True if this store has initialised attributes.'''
        return all([bool(v) for v in vars(self).values()] or [False])


# Keep track of default attribute values of mutable type.
# These always need to be stored per remote node.
defaults = Store()
# In some cases (such as for non-copyable types) a generator is specified
# instead of a default value
generators = list()

# Keep a dict of remote state storage namespaces
remotes = defaultdict(_genstore)


def reset(remote_id=None):
    ''' Reset data for remote '''
    remotes[remote_id or bs.net.act_id] = _genstore()


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
    target = getattr(defaults, group, None) if group else defaults
    if not target:
        return setattr(defaults, group, Store(**{name:default}))
    setattr(target, name, default)


def addgroup(name):
    ''' Add a storage group to each remote data store. '''
    if hasattr(defaults, name):
        return
    # Add store to the defaults
    setattr(defaults, name, Store())

    # Also add to existing stores if necessary
    for remote in remotes.values():
        setattr(remote, name, Store())


def _generator(store, name, objtype, args, kwargs, group=None):
    setattr(getattr(store, group) if group else store, name, objtype(*args, **kwargs))


class ActData:
    ''' Access data from the active remote as if it is a member variable. '''
    __slots__ = ('default', 'name', 'group')

    def __init__(self, *args, name='', group=None, **kwargs):
        self.default = (args, kwargs)
        self.name = name
        self.group = group

    def __set_name__(self, owner, name):
        if not self.name:
            # Get name from attribute name if not previously specified
            self.name = name
        args, kwargs = self.default
        # Retrieve annotated object type if present
        objtype = owner.__annotations__.get(name)
        if objtype:
            self.default = objtype(*args, **kwargs)
        elif len(args) != 1:
            raise AttributeError('A default value and/or a type annotation should be provided with ActData')
        else:
            self.default = args[0]

        # If underlying datatype is mutable, always immediately
        # store per remote node
        if not isinstance(self.default, (str, tuple, Number, frozenset, bytes)):
            # If an annotated object type is specified create a generator for it
            if objtype:                
                generators.append(partial(_generator, name=name, objtype=objtype, args=args, kwargs=kwargs, group=self.group))
                # Add group if it doesn't exist yet
                if self.group is not None:
                    addgroup(self.group)

                # In case remote data already exists, update stores
                for remote_id in remotes.keys():
                    setvalue(name, objtype(*args, **kwargs), remote_id, self.group)
            # Otherwise assume deepcopy can be used to generate initial values per remote
            else:
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
