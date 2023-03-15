''' BlueSky shared state class.

    This class is used to keep a shared state across client(s) and simulation node(s)
'''
from collections import defaultdict
from enum import Enum
import numpy as np
from typing import Dict, Callable, Optional

import bluesky as bs
from bluesky.core import signal, remotestore as rs
from bluesky.core.funcobject import FuncObject
from bluesky.core.timedfunction import timed_function
from bluesky.core.walltime import Timer
from bluesky.network import context as ctx
from bluesky.network.subscription import Subscription, subscriber as nwsub

#TODO: trigger voor actnode changed?


# Keep track of the set of subscribed topics. Store signals to emit
# whenever a state update of each topic is received
changed: Dict[str, signal.Signal] = dict()


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
    ActChange = b'C'


@nwsub
def reset(*args):
    ''' Process incoming RESET events. '''
    rs.reset(ctx.sender_id)
    if ctx.sender_id == bs.net.act_id:
        ctx.action = ActionType.Reset
        ctx.action_content = None
        for topic, sig in changed.items():
            store = rs.get(group=topic.lower())
            sig.emit(store)
        ctx.action = None


@signal.subscriber(topic='actnode-changed')
def on_actnode_changed(act_id):
    ctx.action = ActionType.ActChange
    ctx.action_content = None
    for topic, sig in changed.items():
            store = rs.get(group=topic.lower())
            sig.emit(store)
    ctx.action = None


@signal.subscriber(topic='node-added')
def on_node_added(node_id):
    bs.net.send('REQUEST', list(changed.keys()), to_group=node_id)

def receive(action, data):
    ''' Retrieve and process state data. '''
    store = rs.get(ctx.sender_id, ctx.topic.lower())

    # Store sharedstate context
    ctx.action = ActionType(action)
    ctx.action_content = data

    if ctx.action == ActionType.Update:
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
    elif ctx.action == ActionType.Delete:
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

    elif ctx.action == ActionType.Append:
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

    elif ctx.action == ActionType.Replace:
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
        if itopic not in changed:
            # Subscribe to this network topic
            Subscription(itopic, actonly=actonly).connect(receive)

            # Add data store default to actdata
            rs.addgroup(itopic.lower())

            # Create the signal to emit whenever data of the active remote changes
            changed[itopic] = signal.Signal(f'state-changed.{itopic.lower()}')
        changed[itopic].connect(ifunc)
        return func

    # Allow both @subscriber and @subscriber(args)
    return deco if func is None else deco(func)

def send_update(topic, to_group='', **data):
    bs.net.send(topic, [ActionType.Update.value, data], to_group)


def send_delete(topic, to_group='', **keys):
    bs.net.send(topic, [ActionType.Delete.value, keys], to_group)


def send_append(topic, to_group='', **data):
    bs.net.send(topic, [ActionType.Append.value, data], to_group)


def send_replace(topic, to_group='', **data):
    bs.net.send(topic, [ActionType.Replace.value, data], to_group)


class PublisherMeta(type):
    __publishers__ = dict()
    __timers__ = dict()

    def __call__(cls, topic: str, dt=None, collect=False):
        pub = PublisherMeta.__publishers__.get(topic)
        if pub is None:
            pub = PublisherMeta.__publishers__[topic] = super().__call__(topic, dt, collect)
            # If dt is specified, also create a timer
            if dt is not None:
                if dt not in PublisherMeta.__timers__:
                    timer = PublisherMeta.__timers__[dt] = Timer(dt)
                else:
                    timer = PublisherMeta.__timers__[dt]
                timer.timeout.connect(pub.send_replace)
        return pub

    @nwsub
    @staticmethod
    def request(*topics):
        for pub in (p for p in map(PublisherMeta.__publishers__.get, topics) if p is not None):
            pub.send_replace(to_group=ctx.sender_id)


class Publisher(metaclass=PublisherMeta):
    __collect__ = defaultdict(list)

    @classmethod
    def collect(cls, topic: str, payload: list, to_group: str=''):
        if payload[0] == ActionType.Replace:
            cls.__collect__[(topic, to_group)] = payload
            return
        store = cls.__collect__[(topic, to_group)]
        if not store or store[-2] not in (payload[0], ActionType.Replace.value):
            store.extend(payload)
        elif payload[0] == ActionType.Update.value:
            rec_update(store[1], payload[1])
        elif payload[0] == ActionType.Append.value:
            for key, item in payload[1].items():
                pass

    @staticmethod
    @timed_function(hook='update')
    def send_collected():
        while Publisher.__collect__:
            (topic, to_group), payload = Publisher.__collect__.popitem()
            bs.net.send(topic, payload, to_group)

    def __init__(self, topic: str, dt=None, collect=False) -> None:
        self.topic = topic
        self.dt = dt
        self.collects = collect


    def send_update(self, to_group=b'', **data):
        if data:
            if self.collects:
                self.collect(self.topic, [ActionType.Update.value, data], to_group)
            else:
                bs.net.send(self.topic, [ActionType.Update.value, data], to_group)


    def send_delete(self, to_group=b'', **keys):
        if keys:
            bs.net.send(self.topic, [ActionType.Delete.value, keys], to_group)


    def send_append(self, to_group=b'', **data):
        if data:
            bs.net.send(self.topic, [ActionType.Append.value, data], to_group)


    def send_replace(self, to_group=b'', **data):
        data = data or self.get_payload()
        if data:
            bs.net.send(self.topic, [ActionType.Replace.value, data], to_group)

    def payload(self, func: Callable):
        ''' Decorator method to specify payload getter for this Publisher. '''
        self.get_payload = FuncObject(func)
        return func


# Publisher decorator?
def publisher(func: Optional[Callable] = None, *, topic='', dt=None):
    def deco(func):
        ifunc = func.__func__ if isinstance(func, (staticmethod, classmethod)) \
            else func
        itopic = (topic or ifunc.__name__).upper()

        Publisher(itopic, dt).payload(func)
        return func

    # Allow both @publisher and @publisher(args)
    return deco if func is None else deco(func)