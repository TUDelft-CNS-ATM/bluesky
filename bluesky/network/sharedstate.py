''' BlueSky shared state classes and functions.

    BlueSky's sharedstate is used to keep a shared state 
    across client(s) and simulation node(s)
'''
from typing import Any, Generic, Optional, Type, TypeVar
import numpy as np

from types import SimpleNamespace
from copy import deepcopy
from collections import defaultdict

import bluesky as bs
from bluesky.core import signal
from bluesky.network import context as ctx
from bluesky.network.common import ActionType


# Keep track of the set of subscribed sharedstate topics. Store signals to emit
# whenever a state update of each topic is received
sigchanged: dict[str, signal.Signal] = dict()


def reset(remote_id=None):
    ''' Reset shared state data to defaults for remote simulation. '''
    remotes[remote_id or bs.net.act_id] = _genstore()

    # If this is the active node, also emit a signal about this change
    if ctx.sender_id == bs.net.act_id:
        ctx.action = ActionType.Reset
        ctx.action_content = None
        for topic, sig in sigchanged.items():
            store = get(group=topic.lower())
            sig.emit(store)
        ctx.action = ActionType.NoAction


@signal.subscriber(topic='actnode-changed')
def on_actnode_changed(act_id):
    ctx.action = ActionType.ActChange
    ctx.action_content = None
    for topic, sig in sigchanged.items():
            store = get(group=topic.lower())
            sig.emit(store)
    ctx.action = ActionType.NoAction


def on_sharedstate_received(action, data):
    ''' Retrieve and process state data. '''
    store = get(ctx.sender_id, ctx.topic.lower())

    # Store sharedstate context
    ctx.action = ActionType(action)
    ctx.action_content = data

    if ctx.action == ActionType.Update:
        store.update(data)

    elif ctx.action == ActionType.Append:
        store.append(data)

    elif ctx.action == ActionType.Extend:
        store.extend(data)

    elif ctx.action == ActionType.Replace:
        store.replace(data)

    elif ctx.action == ActionType.Delete:
        store.delete(data)

    # Inform subscribers of state update
    # TODO: what to do with act vs all?
    if ctx.sender_id == bs.net.act_id:
        sigchanged[ctx.topic.lower()].emit(store)

    # Reset context variables
    ctx.action = ActionType.NoAction
    ctx.action_content = None


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


def setdefault(name: str, default: Any, group: Optional[str]=None):
    ''' Set the default value for variable 'name' in group 'group' '''
    # Add group if it doesn't exist yet
    if group is not None:
        addtopic(group)

    target = getattr(defaults, group) if group else defaults
    if not hasattr(target, name):
        # In case remote data already exists and this is a previously unknown variable, update stores
        for remote_id in remotes.keys():
            setvalue(name, deepcopy(default), remote_id, group)
    setattr(target, name, default)


def addtopic(topic: str) -> signal.Signal:
    ''' Add a sharedstate topic if it doesn't yet exist.
    
        This creates a storage group for this topic, which is added to each 
        remote data store.
        
        Arguments:
        - topic:  The sharedstate topic to add
    '''
    topic = topic.lower()

    # Create/get the signal that is emitted when this data changes
    sig = signal.Signal(f'state-changed.{topic}')
    sigchanged[topic] = sig

    # No creation needed if topic is already known
    if not hasattr(defaults, topic):
        # Add store to the defaults
        setattr(defaults, topic, Store())

        # Also add to existing stores if necessary
        for remote in remotes.values():
            setattr(remote, topic, Store())

    return sig


def is_sharedstate(topic):
    ''' Inform whether 'topic' is (already) known as sharedstate. '''
    return topic.lower() in sigchanged


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


# Type template variable for ActData accessor class
T = TypeVar('T')

class ActData(Generic[T]):
    ''' Access shared state data from the active remote as if it is a member variable. '''
    __slots__ = ('default', 'name', 'group')

    def __init__(self, default=None, name='', group=None):
        self.default = default
        self.name: str = name
        self.group: Optional[str] = group

    def __class_getitem__(cls, key):
        ''' Make sure that the final annotation is equal to the wrapped type '''
        return key

    def __set_name__(self, owner, name):
        if not self.name:
            # Get name from attribute name if not previously specified
            self.name = name

        # If we have a default value, and it is not centrally known yet
        # store it for Store generation
        store = defaults if self.group is None else getattr(defaults, self.group, None)
        if store is None or not hasattr(store, name):
            if self.default is not None:
                # If specified use our default
                setdefault(name, self.default, self.group)
            else:
                # Otherwise try to get the default type from annotation
                # Look for __origin__ in case we have a GenericAlias like list[int]
                tp = owner.__annotations__.get(name)
                tp = getattr(tp, '__origin__', tp)
                if isinstance(tp, Type):
                    # Exception case for NumPy ndarray,
                    # which has shape as a mandatory ctor argument
                    if tp is np.ndarray:
                        setdefault(name, np.ndarray(0), self.group)
                    else:
                        setdefault(name, tp(), self.group)

    def __get__(self, obj, objtype=None) -> T:
        ''' Return the actual value for the currently active node. '''
        store = remotes[bs.net.act_id] if bs.net.act_id else defaults
        return getattr(store if self.group is None else getattr(store, self.group), self.name)
        # raise KeyError(f'ActData: {self.name} not found, group={self.group}, active node id={bs.net.act_id}')
        # TODO: What is the active (client) node on the sim-side? Is this always within a currently processed stack command? -> stack.sender_id

    def __set__(self, obj, value: T):
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
    return store


# Keep track of default attribute values of mutable type.
# These always need to be stored per remote node.
defaults = Store()

# Keep a dict of remote state storage namespaces
remotes = defaultdict(_genstore)
