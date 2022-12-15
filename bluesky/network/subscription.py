from bluesky.core.signal import Signal, SignalFactory
import bluesky as bs


class SubscriptionFactory(SignalFactory):
    # Individually keep track of all network subscriptions
    subscriptions = dict()

    def __call__(cls, topic, from_id=None, to_group=None, actonly=None):
        ''' Factory function for Signal construction. '''
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

        if from_id or to_group is not None:
            sub.requested.append((from_id, to_group))
        if actonly is not None:
            sub.actonly = actonly
        return sub


class Subscription(Signal, metaclass=SubscriptionFactory):
    def __init__(self, topic, from_id='', to_group=None, actonly=False):
        super().__init__(topic)
        self.subs = set()
        self.requested = set()
        if from_id or to_group is not None:
            self.requested.append((from_id, to_group))
        self.actonly = actonly

    @property
    def active(self):
        return bool(self.subs)

    def connect(self, func):
        self.subscribe_all()
        return super().connect(func)

    def subscribe(self, from_id='', to_group=None):
        if (from_id or to_group is not None) and (from_id, to_group) not in self.subs:
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


def subscriber(func=None, topic='', **kwargs):
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
        func = func.__func__ if isinstance(func, (staticmethod, classmethod)) \
            else func
        
        # Create the subscription object. Network subscriptions will be made as
        # soon as the network connection is available
        Subscription(topic or func.__name__.upper(), **kwargs).connect(func)

        # Construct the subscription object, but return the original function
        return func
    # Allow both @command and @command(args)
    return deco if func is None else deco(func)
