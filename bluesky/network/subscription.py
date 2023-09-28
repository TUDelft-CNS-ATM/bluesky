from typing import Dict
from bluesky.core.signal import Signal, SignalFactory
from bluesky.network.common import GROUPID_DEFAULT
import bluesky as bs


class SubscriptionFactory(SignalFactory):
    # Individually keep track of all network subscriptions
    subscriptions: Dict[str, 'Subscription'] = dict()

    def __call__(cls, topic, from_id='', to_group=GROUPID_DEFAULT, actonly=None, directedonly=False):
        ''' Factory function for Signal construction. 
        
            Arguments:
            - topic: The topic to subscribe to
            - from_id: When specified, only subscribe to the specified topic transmissions from a single origin.
            - to_group: Subscribe to messages targeted at a specific group. By default, a subscription is
              made to the top-level group (Simulation/Client group)
            - actonly: When set to true, only messages from the currently active remote are subscribed to.
              When the active remote is changed, all actonly subscriptions are automatically adjusted to this 
              new active node.
            - directedonly: Only subscribe to messages specifically directed at this network node.
        '''
        # # Convert name to string if necessary
        if isinstance(topic, bytes):
            topic = topic.decode()
        sub = SubscriptionFactory.subscriptions.get(topic)
        if sub is None:
            sub = super().__call__(topic)

            # Check if type is correct. It could be that Subscription was previously
            # initialised as plain Signal
            if not isinstance(sub, Subscription):
                if issubclass(Subscription, type(sub)):
                    # Signal object instance needs to stay intact, so instead change type and reinitialise
                    sub.__class__ = Subscription
                    sub.subs = set()
                    sub.requested = set()
                else:
                    raise TypeError(f'Trying to connect network subscription to signal with incompatible type {type(sub).__name__}')

            # Store subscription
            SubscriptionFactory.subscriptions[topic] = sub

        if not directedonly and (from_id, to_group) not in sub.subs:
            sub.requested.add((from_id, to_group))
        if actonly is not None:
            sub.actonly = actonly
        return sub


class Subscription(Signal, metaclass=SubscriptionFactory):
    def __init__(self, topic):
        super().__init__(topic)
        self.subs = set()
        self.requested = set()
        self.actonly = False

    @property
    def active(self):
        return bool(self.subs)

    def connect(self, func):
        self.subscribe_all()
        return super().connect(func)

    def subscribe(self, from_id='', to_group=GROUPID_DEFAULT):
        if (from_id, to_group) not in self.subs:
            if bs.net is not None:
                self.subs.add((from_id, to_group))
                if self.actonly:
                    bs.net._subscribe(self.topic, from_id, to_group, self.actonly)
                else:
                    bs.net._subscribe(self.topic, from_id, to_group)
            else:
                self.requested.add((from_id, to_group))

    def subscribe_all(self):
        if bs.net is not None:
            while self.requested:
                self.subscribe(*self.requested.pop())

    def unsubscribe(self, from_id='', to_group=None):
        if (from_id or to_group is not None):
            to_group = to_group or GROUPID_DEFAULT
            if (from_id, to_group) in self.subs:
                self.subs.discard((from_id, to_group))
                if bs.net is not None:
                    bs.net._unsubscribe(self.topic, from_id, to_group)
        else:
            # Unsubscribe all
            self.requested.clear()
            if bs.net is not None:
                while self.subs:
                    bs.net._unsubscribe(self.topic, *self.subs.pop())


def subscriber(func=None, *, topic='', directedonly=False, **kwargs):
    ''' BlueSky network subscription decorator.

        Functions decorated with this decorator will be called whenever data
        with the specified topic is received.

        Arguments:
        - topic: The topic to subscribe to for this function
        - from_id: Subscribe to data from a specific sender (optional)
        - to_group: Subscribe to data sent to a specific group (optional)
        - actonly: Only receive this data for the active node (client only)
    '''
    def deco(func):
        ifunc = func.__func__ if isinstance(func, (staticmethod, classmethod)) \
            else func
        
        # Create the subscription object. Network subscriptions will be made as
        # soon as the network connection is available
        Subscription(topic or ifunc.__name__.upper(), directedonly=directedonly, **kwargs).connect(ifunc)

        # Construct the subscription object, but return the original function
        return func
    # Allow both @subscriber and @subscriber(args)
    return deco if func is None else deco(func)
