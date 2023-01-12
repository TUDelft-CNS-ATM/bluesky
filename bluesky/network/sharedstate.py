''' BlueSky shared state class.

    This class is used to keep a shared state across client(s) and simulation node(s)
'''
from types import SimpleNamespace
from collections import defaultdict
import numpy as np

import bluesky as bs
from bluesky.core import signal
from bluesky.network.subscription import Subscription

#TODO: trigger voor actnode changed?

# Keep a dict of remote state storage namespaces
remotes = defaultdict(lambda: SimpleNamespace())

# Keep track of the set of subscribed topics
topics = set()

# Signal to emit whenever a state update is received
changed = signal.Signal('state-changed')


def get(remote_id):
    return remotes[remote_id]


def receive(data):
    ''' Retrieve and process state data. '''
    remote = remotes[bs.net.sender_id]

    action, actdata = data
    if action == b'U':
        # Update
        for key, item in actdata.items():
            container = getattr(remote, key)
            if isinstance(container, dict):
                container.update(item)
            else:
                for idx, value in item.items():
                    container[idx] = value
    elif action == b'D':
        # Delete
        for key, item in actdata.items():
            container = getattr(remote, key)
            if isinstance(container, np.ndarray):
                mask = np.ones_like(container, dtype=bool)
                mask[item] = False
                setattr(remote, key, container[mask])
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
        for key, item in actdata.items():
            if isinstance(item, list):
                getattr(remote, key).extend(item)
            else:
                getattr(remote, key).append(item)

    elif action == b'F':
        # Full replace
        vars(remote).update(actdata)

    # Inform subscribers of state update
    topic = bs.net.topic.decode()
    changed[topic].emit(remote)


def subscriber(func=None, *, topic='', actonly=False):
    ''' Subscribe to a state topic. '''
    def deco(func):
        ifunc = func.__func__ if isinstance(func, (staticmethod, classmethod)) \
            else func

        itopic = topic or ifunc.__name__.upper()
        # Create a new network subscription if 
        if itopic not in topics:
            Subscription(itopic, actonly=actonly).connect(receive)
            topics.add(itopic)
        changed[itopic].connect(ifunc)
        return func

    # Allow both @subscriber and @subscriber(args)
    return deco if func is None else deco(func)

def send_update(topic, **data):
    bs.net.send(topic, [b'U', data])


def send_delete(topic, **keys):
    bs.net.send(topic, [b'D', keys])


def send_append(topic, **data):
    bs.net.send(topic, [b'A', data])


def send_full(topic, **data):
    bs.net.send(topic, [b'F', data])
