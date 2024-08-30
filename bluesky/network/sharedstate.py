''' BlueSky shared state classes and functions.

    BlueSky's sharedstate is used to keep a shared state 
    across client(s) and simulation node(s)
'''
import inspect
import numpy as np
from numbers import Number
from functools import partial
from types import SimpleNamespace
from copy import deepcopy
from collections import defaultdict

import bluesky as bs
from bluesky.network import context as ctx


def reset(remote_id=None):
    ''' Reset shared state data to defaults for remote simulation. '''
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


def addtopic(topic):
    ''' Add a sharedstate topic if it doesn't yet exist.
    
        This creates a storage group for this topic, which is added to each 
        remote data store.
        
        Arguments:
        - topic:  The sharedstate topic to add
    '''
    topic = topic.lower()
    # No creation needed if topic is already known
    if hasattr(defaults, topic):
        return

    # Add store to the defaults
    setattr(defaults, topic, Store())

    # Also add to existing stores if necessary
    for remote in remotes.values():
        setattr(remote, topic, Store())


class Store(SimpleNamespace):
    ''' Simple storage object for nested storage of state data per simulation node. '''
    def valid(self):
        ''' Return True if this store has initialised attributes.'''
        return all([bool(v) for v in vars(self).values()] or [False])

    def update(self, data):
        ''' Update a value in this store. '''
        for key, item in data.items():
            container = getattr(self, key, None)
            if container is None:
                setattr(self, key, item)
            elif isinstance(container, dict):
                _recursive_update(container, item)
            else:
                for idx, value in item.items():
                    container[idx] = value

    def append(self, data):
        ''' Append data to (lists/arrays in) this store. '''
        for key, item in data.items():
            container = getattr(self, key, None)
            if container is None:
                setattr(self, key, [item])
            elif isinstance(container, np.ndarray):
                setattr(self, key, np.append(container, item))

    def extend(self, data):
        ''' Extend data in (lists/arrays in) this store. '''
        for key, item in data.items():
            container = getattr(self, key, None)
            if container is None:
                setattr(self, key, item)
            elif isinstance(container, list):
                container.extend(item)
            elif isinstance(container, np.ndarray):
                setattr(self, key, np.concatenate([container, item]))

    def replace(self, data):
        ''' Replace data containers in this store. '''
        vars(self).update(data)

    def delete(self, data):
        ''' Delete data from this store. '''
        # We are expecting either an index, or a key value from a reference variable
        for key, item in data.items():
            idx = None
            if key not in vars(self):
                # Assume we are receiving an index to arrays/lists in this store
                if not isinstance(item, int):
                    raise ValueError(f"Expected integer index for delete {key} in topic {ctx.topic}") 
                idx = item
            else:
                ref = getattr(self, key)
                # If ref is a dict, this delete action should only act on the dict.
                if isinstance(ref, dict):
                    if isinstance(item, (list, tuple)):
                        for key in item:
                            ref.pop(key, None)
                    else:
                        ref.pop(item, None)
                    continue
                # Otherwise, assume we are receiving a lookup key for a reference value
                elif isinstance(ref, np.ndarray):
                    indices = np.where(ref == item)[0]
                    if not indices:
                        raise KeyError(f'Item with key {item} not found for variable {key} in topic {ctx.topic}')
                    idx = indices[0] if len(indices) == 1 else indices
                elif isinstance(ref, list):
                    idx = ref.index(item)

            if idx is None:
                continue

            for container in vars(self).values():
                if isinstance(container, np.ndarray):
                    mask = np.ones_like(container, dtype=bool)
                    mask[idx] = False
                    setattr(self, key, container[mask])
                elif isinstance(container, list):
                    if isinstance(idx, int):
                        container.pop(idx)
                    else:
                        # Assume a tuple or list
                        for iidx in reversed(sorted(idx)):
                            container.pop(iidx)


class ActData:
    ''' Access shared state data from the active remote as if it is a member variable. '''
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
                    addtopic(self.group)

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


def _recursive_update(target, source):
    ''' Recursively update nested dicts/lists/arrays in a Store. '''
    for k, v in source.items():
        if isinstance(v, dict):
            inner = target.get(k)
            if inner is not None:
                _recursive_update(inner, v)
                continue
        target[k] = v


def _genstore():
    ''' Generate a store object for a remote simulation from defaults. '''
    store = deepcopy(defaults)
    for g in generators:
        g(store)
    return store


def _generator(store, name, objtype, args, kwargs, group=None):
    ''' Custom generator for non-base types. '''
    setattr(getattr(store, group) if group else store, name, objtype(*args, **kwargs))


# Keep track of default attribute values of mutable type.
# These always need to be stored per remote node.
defaults = Store()
# In some cases (such as for non-copyable types) a generator is specified
# instead of a default value
generators = list()

# Keep a dict of remote state storage namespaces
remotes = defaultdict(_genstore)
