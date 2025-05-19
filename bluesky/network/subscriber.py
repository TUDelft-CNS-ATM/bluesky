import inspect

import bluesky as bs
from bluesky.core.funcobject import FuncObject
from bluesky.core import signal
import bluesky.network.sharedstate as ss
import bluesky.network.context as ctx
from bluesky.network.common import GROUPID_DEFAULT, ActionType, MessageType


#TODO:
# trigger voor actnode changed?


def subscriber(func=None, *, topic='', broadcast=True, actonly=False, raw=False, from_group=GROUPID_DEFAULT, to_group=''):
    ''' BlueSky network subscription decorator.

        Functions decorated with this decorator will be called whenever data
        with the specified topic is received.

        Arguments:
        - topic:      The topic to subscribe to for this function. Function name
                      is used when no topic is specified
        - broadcast:  Whether to subscribe to broadcast-to-(all/group) topic.
                      When set to False, to_group and from_group are ignored,
                      and only messages sent directly to this client/node are
                      received.
        - raw:        Set to true for SharedState messages if you want to
                      receive the original unprocessed message
        - from_group: Subscribe to data from a specific sender(-group) (optional)
                      By default, broadcast subscriptions filter so that only
                      sim->client and client->sim messages are received (to avoid
                      receiving our own broadcasts). If you don't want this,
                      set from_group='*'.
        - to_group:   Subscribe to data sent to a specific group (optional)
        - actonly:    Only receive this data for the active node (client only)
    '''
    def deco(func):
        # Unwrap wrapped functions, static and class methods
        ifunc = inspect.unwrap(func, stop=lambda f:not isinstance(f, (staticmethod, classmethod)))
        
        # Create the subscription object. Network subscriptions will be made as
        # soon as the network connection is available
        Subscription(topic.upper() or ifunc.__name__.upper(), broadcast, actonly, from_group, to_group).connect(ifunc, raw)

        # Construct the subscription object, but return the original function
        return func
    # Allow both @subscriber and @subscriber(args)
    return deco if func is None else deco(func)


def subscribe(topic, *, broadcast=True, actonly=False, from_group=GROUPID_DEFAULT, to_group='') -> 'Subscription':
    ''' Subscribe to a network topic without passing a function callback. 
    
        Arguments:
        - topic: The topic to subscribe to for this function.
        - broadcast:  Whether to subscribe to broadcast-to-(all/group) topic.
                      When set to False, to_group and from_group are ignored,
                      and only messages sent directly to this client/node are
                      received.
        - from_group: Subscribe to data from a specific sender(-group) (optional)
                      By default, broadcast subscriptions filter so that only
                      sim->client and client->sim messages are received (to avoid
                      receiving our own broadcasts). If you don't want this,
                      set from_group='*'.
        - to_group:   Subscribe to data sent to a specific group (optional)
        - actonly:    Only receive this data for the active node (client only)

        Returns:
        - The subscription object for this topic
    '''
    # Create the subscription object. Network subscriptions will be made as
    # soon as the network connection is available
    return Subscription(topic.upper(), broadcast, actonly, from_group, to_group)


class SubscriptionFactory(signal.SignalFactory):
    # Individually keep track of all network subscriptions
    subscriptions: dict[str, 'Subscription'] = dict()

    def __call__(cls, topic, broadcast=True, actonly=None, from_group=GROUPID_DEFAULT, to_group=''):
        ''' Factory function for Subscription construction. 
        
            Arguments:
            - topic:      The topic to subscribe to for this function. Function name
                          is used when no topic is specified
            - broadcast:  Whether to subscribe to broadcast-to-(all/group) topic.
                          When set to False, to_group and from_group are ignored,
                          and only messages sent directly to this client/node are
                          received.
            - from_group: Subscribe to data from a specific sender(-group) (optional)
                          By default, broadcast subscriptions filter so that only
                          sim->client and client->sim messages are received (to avoid
                          receiving our own broadcasts). If you don't want this,
                          set from_group='*'.
            - to_group:   Subscribe to data sent to a specific group (optional)
            - actonly:    Only receive this data for the active node (client only)
        '''
        # # Convert name to string if necessary
        if isinstance(topic, bytes):
            topic = topic.decode()
        # Get subscription object if it already exists
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
                    sub.actonly = False
                    sub.msg_type = MessageType.Unknown
                    sub.deferred_subs = []
                    signal.Signal.connect(sub, sub._detect_type)
                else:
                    raise TypeError(f'Trying to connect network subscription to signal with incompatible type {type(sub).__name__}')

            # Store subscription
            SubscriptionFactory.subscriptions[topic] = sub

        # Keep track of all broadcast topic orig/dest combinations
        if actonly is not None:
            sub.actonly = actonly
        if broadcast or sub.actonly:
            sub.subscribe(from_group, to_group)

        return sub


class Subscription(signal.Signal, metaclass=SubscriptionFactory):
    ''' Network subscription class.
    
        Objects of this type are used to store subscription details, and to 
        keep track of local topic subscribers
    '''
    def __init__(self, topic, broadcast=True, actonly=False, from_group=GROUPID_DEFAULT, to_group=''):
        super().__init__(topic)
        self.subs = set()
        self.requested = set()
        self.actonly = actonly
        self.deferred_subs = []
        if ss.is_sharedstate(topic):
            # We already know this is a sharedstate topic. subscribe as such
            self.msg_type = MessageType.SharedState
            super().connect(ss.on_sharedstate_received)
        else:
            # Start out uninitialised so we can detect whether incoming data is
            # a shared state or a regular message
            self.msg_type = MessageType.Unknown
            super().connect(self._detect_type)

    def _detect_type(self, *args, **kwargs):
        # First disconnect this function, it is only needed once
        super().disconnect(self._detect_type)
        # This function responds to the first incoming message for topic
        # Detect whether it is a sharedstate message
        if args and ActionType.isaction(args[0]):
            # This is a sharedstate message
            self.msg_type = MessageType.SharedState
            # In this case, all (non-raw) subscribers will be configured
            # as sharedstate subscribers
            sig = ss.addtopic(self.topic)
            while self.deferred_subs:
                sig.connect(self.deferred_subs.pop())

            # Finally send the sharedstate message on to the subscribers,
            # and subscribe the sharedstate processing function to this topic
            super().connect(ss.on_sharedstate_received)
            ss.on_sharedstate_received(*args, **kwargs)

        else:
            self.msg_type = MessageType.Regular
            while self.deferred_subs:
                cb = self.deferred_subs.pop()
                cb(*args, **kwargs)
                super().connect(cb)


    @property
    def active(self):
        ''' Returns True if this Subscription has activated network subscriptions. '''
        return bool(self.subs)

    def connect(self, func, raw=False):
        ''' Connect a callback function to incoming data on this
            subscription topic.
            
            Arguments:
            - func: The callback function
            - raw:  Set this to True for SharedState messages if you want to
                    receive the original unprocessed message
        '''
        self.subscribe_all()
        if raw or self.msg_type == MessageType.Regular:
            super().connect(func)
        elif self.msg_type == MessageType.Unknown:
            self.deferred_subs.append(FuncObject(func))
        else:
            signal.Signal(f'state-changed.{self.topic.lower()}').connect(func)

    def subscribe(self, from_group=GROUPID_DEFAULT, to_group=''):
        if (from_group, to_group) not in self.subs:
            if bs.net is not None:
                self.subs.add((from_group, to_group))
                if self.actonly:
                    bs.net._subscribe(self.topic, from_group, to_group, self.actonly)
                else:
                    bs.net._subscribe(self.topic, from_group, to_group)
            else:
                self.requested.add((from_group, to_group))

    def subscribe_all(self):
        if bs.net is not None:
            while self.requested:
                self.subscribe(*self.requested.pop())

    def unsubscribe(self, from_group=GROUPID_DEFAULT, to_group=''):
        if (from_group or to_group is not None):
            if (from_group, to_group) in self.subs:
                self.subs.discard((from_group, to_group))
                if bs.net is not None:
                    bs.net._unsubscribe(self.topic, from_group, to_group)
        else:
            # Unsubscribe all
            self.requested.clear()
            if bs.net is not None:
                while self.subs:
                    bs.net._unsubscribe(self.topic, *self.subs.pop())


@subscriber
def reset(*args):
    ''' Process incoming RESET events. '''
    # Clear state data to defaults for this simulation node
    ss.reset(ctx.sender_id)


@signal.subscriber(topic='node-added')
def on_node_added(node_id):
    ''' When a new node is announced, request the initial/current state of all 
        subscribed shared states.
    '''
    topics = [topic for topic, sub in SubscriptionFactory.subscriptions.items() 
              if sub.msg_type in (MessageType.Unknown, MessageType.SharedState)]
    bs.net.send('REQUEST', topics, to_group=node_id)
