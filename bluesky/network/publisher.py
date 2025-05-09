import inspect
from collections import defaultdict
from collections.abc import Callable

import bluesky as bs
from bluesky.core.funcobject import FuncObject
from bluesky.core.timedfunction import timed_function
from bluesky.core.walltime import Timer
from bluesky.network.common import ActionType
from bluesky.network.subscriber import subscriber
from bluesky.network.sharedstate import _recursive_update
import bluesky.network.context as ctx


class PublisherMeta(type):
    __publishers__ = dict()
    __timers__ = dict()

    def __call__(cls, topic: str, dt=None, collect=False, send_type='replace'):
        pub = PublisherMeta.__publishers__.get(topic)
        if pub is None:
            pub = PublisherMeta.__publishers__[topic] = super().__call__(topic, dt, collect)
            # If dt is specified, also create a timer
            if dt is not None:
                if dt not in PublisherMeta.__timers__:
                    timer = PublisherMeta.__timers__[dt] = Timer(dt)
                else:
                    timer = PublisherMeta.__timers__[dt]
                callback = getattr(pub, f'send_{send_type}')
                timer.timeout.connect(callback)
        return pub

    @subscriber
    @staticmethod
    def request(*topics):
        ''' Clients can request a full state update, to which this function responds. '''
        for pub in (p for p in map(PublisherMeta.__publishers__.get, topics) if p is not None):
            pub.send_replace(to_group=ctx.sender_id)


class StatePublisher(metaclass=PublisherMeta):
    ''' BlueSky shared state publisher class.

        Use this class if you want more control over publishing shared state data.
        Unlike the state_publisher decorator this class allows you to also send 
        update, delete, append, and extend actions.
    '''
    __collect__ = defaultdict(list)

    @classmethod
    def collect(cls, topic: str, payload: list, to_group: str=''):
        if payload[0] == ActionType.Replace:
            cls.__collect__[(topic, to_group)] = payload
            return
        store = cls.__collect__[(topic, to_group)]
        if not store or store[-2] not in (payload[0], ActionType.Replace.value, ActionType.Extend.value):
            if payload[0] == ActionType.Append.value:
                payload[0] = ActionType.Extend.value
                payload[1] = {k:[v] for k, v in payload[1].items()}
            store.extend(payload)
        elif payload[0] == ActionType.Update.value:
            _recursive_update(store[1], payload[1])
        elif payload[0] == ActionType.Append.value:
            for key, item in payload[1].items():
                store[1][key].append(item)
        elif payload[0] == ActionType.Extend.value:
            for key, item in payload[1].items():
                store[1][key].extend(item)

    @staticmethod
    @timed_function(hook=('update', 'hold'))
    def send_collected():
        while StatePublisher.__collect__:
            (topic, to_group), payload = StatePublisher.__collect__.popitem()
            bs.net.send(topic, payload, to_group)

    def __init__(self, topic: str, dt=None, collect=False, send_type='replace') -> None:
        self.topic = topic
        self.dt = dt
        self.collects = collect

    @staticmethod
    def get_payload():
        ''' Default payload function returns None '''
        return

    def send_update(self, to_group=b'', **data):
        data = data or self.get_payload()
        if data:
            if self.collects:
                self.collect(self.topic, [ActionType.Update.value, data], to_group)
            else:
                bs.net.send(self.topic, [ActionType.Update.value, data], to_group)


    def send_delete(self, to_group=b'', **keys):
        if keys:
            bs.net.send(self.topic, [ActionType.Delete.value, keys], to_group)


    def send_append(self, to_group=b'', **data):
        data = data or self.get_payload()
        if data:
            if self.collects:
                self.collect(self.topic, [ActionType.Append.value, data], to_group)
            else:
                bs.net.send(self.topic, [ActionType.Append.value, data], to_group)

    def send_extend(self, to_group=b'', **data):
        data = data or self.get_payload()
        if data:
            if self.collects:
                self.collect(self.topic, [ActionType.Extend.value, data], to_group)
            else:
                bs.net.send(self.topic, [ActionType.Extend.value, data], to_group)

    def send_replace(self, to_group=b'', **data):
        data = data or self.get_payload()
        if data:
            bs.net.send(self.topic, [ActionType.Replace.value, data], to_group)

    def payload(self, func: Callable):
        ''' Decorator method to specify payload getter for this Publisher. '''
        self.get_payload = FuncObject(func)
        return func


def state_publisher(func: Callable|None = None, *, topic='', dt=None, send_type='replace'):
    ''' BlueSky shared state publisher decorator.

        Convenience decorator to create a periodically called state publisher.
        Functions decorated with this decorator will be:
        - periodically called at interval dt
        - called when a subscriber requests a full state

        Decorated function should return a dictionary with all relevant state data.

        Arguments:
        - func: The payload function that returns the data to be published in a dictionary
        - topic: The message topic that clients can subscribe to. If no topic is given, the name of the
          decorated function is used.
        - dt: Time interval between successive periodic broadcasts of this state publisher
        - send_type: Type of update sent by this publisher. This can be:
          - replace (the default): Each state package sent by this publisher fully replaces previous data
          - append: Each state package sent should be appended to the previously received data
          - extend: Each state package sent should be used to extend the previously received data
          - update: Each state package sent should be used to update the previously received data
    '''
    def deco(func):
        ifunc = inspect.unwrap(func, stop=lambda f:not isinstance(f, (staticmethod, classmethod)))
        itopic = (topic or ifunc.__name__).upper()

        StatePublisher(itopic, dt, send_type=send_type).payload(func)
        return func

    # Allow both @publisher and @publisher(args)
    return deco if func is None else deco(func)