""" BlueSky implementation of signals that can trigger one or more functions
    when a signal is emitted. """
from bluesky.core.funcobject import FuncObject


class SignalFactory(type):
    ''' Factory meta-class for Signal objects in BlueSky. '''
    # Store all signal objects in process
    __signals__ = dict()

    def __call__(cls, topic='', *args, **kwargs):
        ''' Factory function for Signal construction. '''
        # if no name is passed, return an anonymous Signal
        if not topic:
            return super().__call__('ANONYMOUS', *args, **kwargs)
        # Convert name to string if necessary
        if isinstance(topic, bytes):
            topic = topic.decode()
        # Signal topics are case-insensitive
        topic = topic.upper()
        # Check if this Signal already exists. If it doesn't, create it.
        sig = SignalFactory.__signals__.get(topic)
        if sig is None:
            sig = super().__call__(topic, *args, **kwargs)
            cls.__signals__[topic] = sig
        return sig


class Signal(metaclass=SignalFactory):
    """ A signal can trigger one or more functions when it is emitted. """
    def __init__(self, topic=''):
        self.topic = topic
        self.subscribers = list()

    def emit(self, *args, **kwargs):
        """ Trigger the registered functions with passed arguments. """
        # Iterate over shallow copy of subscriber list to avoid issues with connects/disconnects
        # that are called within one of our subscribed callback functions
        for sub in list(self.subscribers):
            sub(*args, **kwargs)

    def connect(self, func):
        """ Connect a new function to this signal. """
        self.subscribers.append(FuncObject(func))

    def disconnect(self, func):
        """ Disconnect a function from this signal. """
        try:
            self.subscribers.remove(FuncObject(func))
        except AttributeError:
            print(f'Function {func.__name__} is not connected to a signal.')
        except ValueError:
            print(f'Function {func.__name__} is not connected to Signal {self.topic}.')


def subscriber(func=None, topic=''):
    ''' BlueSky Signal subscription decorator.

        Functions decorated with this decorator will be called whenever the
        corresponding Signal emits data.

        Arguments:
        - topic: The topic to subscribe to for this function
    '''
    def deco(func):
        # Subscribe to topic.
        Signal(topic or func.__name__.upper()).connect(func)

        # Construct the subscription object, but return the original function
        return func
    # Allow both @subscriber and @subscriber(args)
    return deco if func is None else deco(func)
