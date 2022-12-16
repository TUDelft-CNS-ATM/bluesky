""" BlueSky implementation of signals that can trigger one or more functions
    when a signal is emitted. """
import inspect
from dataclasses import dataclass


@dataclass(slots=True)
class Subscriber:
    ''' Functions that are connected to a Signal are stored as Subscribers. '''
    func: callable
    # TODO: add signal to dict in Subscriber

    def notimplemented(self, *args, **kwargs):
        pass

    @property
    def valid(self):
        spec = inspect.signature(self.func)
        # Check if this is an unbound class/instance method
        return spec.parameters.get('self') is None and \
            spec.parameters.get('cls') is None


class SignalFactory(type):
    ''' Factory meta-class for Signal objects in BlueSky. '''
    # Store all signal objects in process
    __signals__ = dict()

    def __call__(cls, topic='', *args, **kwargs):
        ''' Factory function for Signal construction. '''
        # if no name is passed, return an anonymous Signal
        if not topic:
            return super().__call__('anonymous', *args, **kwargs)
        # Convert name to string if necessary
        if isinstance(topic, bytes):
            topic = topic.decode()
        # Check if this Signal already exists
        sig = SignalFactory.__signals__.get(topic)
        if sig is None:
            # If it doesn't, create it
            sig = super().__call__(topic, *args, **kwargs)
            SignalFactory.__signals__[topic] = sig
        return sig


class Signal(metaclass=SignalFactory):
    """ A signal can trigger one or more functions when it is emitted. """
    def __init__(self, topic=''):
        self.topic = topic
        self.subscribers = list()

    def emit(self, *args, **kwargs):
        """ Trigger the registered functions with passed arguments. """
        for sub in self.subscribers:
            sub.func(*args, **kwargs)

    def connect(self, func):
        """ Connect a new function to this signal. """
        if inspect.ismethod(func):
            if not hasattr(func.__func__, '__subscriber__'):
                func.__func__.__subscriber__ = Subscriber(func)
            sub = func.__func__.__subscriber__
        else:
            if not hasattr(func, '__subscriber__'):
                func.__subscriber__ = Subscriber(func)
            sub = func.__subscriber__
        self.subscribers.append(sub)

    def disconnect(self, func):
        """ Disconnect a function from this signal. """
        try:
            sub = func.__func__.__subscriber__ if inspect.ismethod(func) else func.__subscriber__
            self.subscribers.remove(sub)
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
        func = func.__func__ if isinstance(func, (staticmethod, classmethod)) \
            else func
        
        # Subscribe to topic.
        Signal(topic or func.__name__.upper()).connect(func)

        # Construct the subscription object, but return the original function
        return func
    # Allow both @subscriber and @subscriber(args)
    return deco if func is None else deco(func)
