''' BlueSky shared state class.

    This class is used to keep a shared state across client(s) and simulation node(s)
'''
from enum import Enum
import numpy as np

import bluesky as bs
from bluesky.core import signal, remotestore as rs
from bluesky.network import context as ctx
from bluesky.network.subscription import Subscription, subscriber as nwsub

#TODO: trigger voor actnode changed?


# Keep track of the set of subscribed topics
topics = set()

# Signal to emit whenever a state update is received
changed = signal.Signal('state-changed')


class ActionType(Enum):
    ''' Shared state action types. 
    
        An incoming shared state update can be of the following types:
        Append: One or more items are appended to the state
        Delete: One or more items are deleted from the state
        Update: One or more items within the state are updated
        Replace: The full state object is replaced
    '''
    Append = b'A'
    Delete = b'D'
    Update = b'U'
    Replace = b'R'
    Reset = b'X'


@nwsub
def reset(*args):
    ''' Process incoming RESET events. '''
    rs.reset(ctx.sender_id)
    if ctx.sender_id == bs.net.act_id:
        ctx.action = ActionType.Reset
        ctx.action_content = None
        changed.emit(None)
        ctx.action = None


def receive(action, data):
    ''' Retrieve and process state data. '''
    store = rs.get(ctx.sender_id, ctx.topic.lower())

    # Store sharedstate context
    ctx.action = ActionType(action)
    ctx.action_content = data

    if action == b'U':
        # Update
        for key, item in data.items():
            container = getattr(store, key, None)
            if container is None:
                setattr(store, key, item)
            elif isinstance(container, dict):
                # container.update(item)
                rec_update(container, item)
            else:
                for idx, value in item.items():
                    container[idx] = value
    elif action == b'D':
        # Delete
        for key, item in data.items():
            container = getattr(store, key, None)
            if container is None:
                continue
            if isinstance(container, np.ndarray):
                mask = np.ones_like(container, dtype=bool)
                mask[item] = False
                setattr(store, key, container[mask])
            elif isinstance(container, list):
                if isinstance(item, int):
                    container.pop(item)
                else:
                    # Assume a tuple or list
                    for idx in reversed(sorted(item)):
                        container.pop(idx)
            elif isinstance(container, dict):
                if isinstance(item, (list, tuple)):
                    for key in item:
                        container.pop(key, None)
                else:
                    container.pop(item, None)

    elif action == b'A':
        # Append
        # TODO: other types? (ndarray, ...)
        # TODO: first reception is scalar? Allow scalars at all?
        for key, item in data.items():
            container = getattr(store, key, None)
            if container is None:
                setattr(store, key, item if isinstance(item, (list, tuple, np.ndarray)) else [item])
            elif isinstance(item, list):
                container.extend(item)
            else:
                container.append(item)

    elif action == b'R':
        # Full replace
        vars(store).update(data)

    # Inform subscribers of state update
    # TODO: what to do with act vs all?
    if ctx.sender_id == bs.net.act_id:
        changed[ctx.topic].emit(store)

    # Reset context variables
    ctx.action = None
    ctx.action_content = None


def rec_update(target, source):
    for k, v in source.items():
        if isinstance(v, dict):
            inner = target.get(k)
            if inner is not None:
                rec_update(inner, v)
                continue
        target[k] = v


def subscriber(func=None, *, topic='', actonly=False):
    ''' Decorator to subscribe to a state topic. '''
    def deco(func):
        ifunc = func.__func__ if isinstance(func, (staticmethod, classmethod)) \
            else func

        itopic = (topic or ifunc.__name__).upper()
        # Create a new network subscription if 
        if itopic not in topics:
            # Subscribe to this network topic
            Subscription(itopic, actonly=actonly).connect(receive)
            topics.add(itopic)
            # Add data store default to actdata
            rs.addgroup(itopic.lower())
        changed[itopic].connect(ifunc)
        return func

    # Allow both @subscriber and @subscriber(args)
    return deco if func is None else deco(func)

def send_update(topic, to_group='', **data):
    bs.net.send(topic, [b'U', data], to_group)


def send_delete(topic, to_group='', **keys):
    bs.net.send(topic, [b'D', keys], to_group)


def send_append(topic, to_group='', **data):
    bs.net.send(topic, [b'A', data], to_group)


def send_replace(topic, to_group='', **data):
    bs.net.send(topic, [b'R', data], to_group)
