from bluesky.core.signal import Signal, SignalFactory
import bluesky as bs


class SubscriptionFactory(SignalFactory):
    # Individually keep track of all network subscriptions
    subscriptions = dict()

    def __call__(cls, topic, from_id=None, to_group=None, actonly=None, targetonly=None):
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
                else:
                    raise TypeError(f'Trying to connect network subscription to signal with incompatible type {type(sub).__name__}')

            # Store subscription
            SubscriptionFactory.subscriptions[topic] = sub

        if from_id is not None:
            sub.from_id = from_id
        if to_group is not None:
            sub.to_group = to_group
        if actonly is not None:
            sub.actonly = actonly
        if targetonly is not None:
            sub.targetonly = targetonly
        return sub


class Subscription(Signal, metaclass=SubscriptionFactory):
    def __init__(self, topic, from_id='', to_group=None, actonly=False, targetonly=False):
        super().__init__(topic)
        self.from_id = from_id
        self.to_group = to_group
        self.actonly = actonly
        self.targetonly = targetonly


def subscriber(func=None, topic='', targetonly=False, **kwargs):
    ''' BlueSky network subscription decorator.

        Functions decorated with this decorator will be called whenever data
        with the specified topic is received.

        Arguments:
        - topic: The topic to subscribe to for this function
        - from_id: Subscribe to data from a specific sender (optional)
        - to_group: Subscribe to data sent to a specific group (optional)
        - actonly: Only receive this data for the active node (client only)
        - targetonly: Only receive this data if its directed specificly to me (optional)
    '''
    def deco(func):
        func = func.__func__ if isinstance(func, (staticmethod, classmethod)) \
            else func
        
        # If possible immediately subscribe to topic. otherwise just create
        # subscription object
        if bs.net and not targetonly:
            bs.net.subscribe(topic or func.__name__.upper(), **kwargs).connect(func)
        else:
            Subscription(topic or func.__name__.upper(), targetonly=targetonly, **kwargs).connect(func)

        # Construct the subscription object, but return the original function
        return func
    # Allow both @command and @command(args)
    return deco if func is None else deco(func)
