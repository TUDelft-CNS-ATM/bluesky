''' BlueSky shared state class.

    This class is used to keep a shared state across client(s) and simulation node(s)
'''
from types import SimpleNamespace
import numpy as np

import bluesky as bs
from bluesky.core import signal, actdata
from bluesky.network.subscription import Subscription

#TODO: trigger voor actnode changed?


# Keep track of the set of subscribed topics
topics = set()

# Signal to emit whenever a state update is received
changed = signal.Signal('state-changed')


def receive(action, data):
    ''' Retrieve and process state data. '''
    topic = bs.net.topic.decode()
    remote = actdata.remotes[bs.net.sender_id]
    store = getattr(remote, topic.lower())

    if action == b'U':
        # Update
        for key, item in data.items():
            container = getattr(store, key)
            if isinstance(container, dict):
                container.update(item)
            else:
                for idx, value in item.items():
                    container[idx] = value
    elif action == b'D':
        # Delete
        for key, item in data.items():
            container = getattr(store, key)
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

    elif action == b'F':
        # Full replace
        vars(store).update(data)

    # Inform subscribers of state update
    # TODO: what to do with act vs all?
    if bs.net.sender_id == bs.net.act_id:
        changed[topic].emit(store)


def subscriber(func=None, *, topic='', actonly=False):
    ''' Subscribe to a state topic. '''
    def deco(func):
        ifunc = func.__func__ if isinstance(func, (staticmethod, classmethod)) \
            else func

        itopic = topic or ifunc.__name__.upper()
        # Create a new network subscription if 
        if itopic not in topics:
            # Subscribe to this network topic
            Subscription(itopic, actonly=actonly).connect(receive)
            topics.add(itopic)
            # Add data store default to actdata
            actdata.adddefault(itopic.lower(), SimpleNamespace())
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


def send_full(topic, to_group='', **data):
    bs.net.send(topic, [b'F', data], to_group)
