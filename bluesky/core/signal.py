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
            return super().__call__('anonymous', *args, **kwargs)
        # Convert name to string if necessary
        if isinstance(topic, bytes):
            topic = topic.decode()
        # Signal topics are case-insensitive
        topic = topic.upper()
        # Check if this Signal already exists. If it doesn't, create it.
        return SignalFactory.__signals__.get(topic) or \
            cls.__create_sig__(topic, *args, **kwargs)

    def __create_sig__(cls, topic, *args, **kwargs):
        # First check if a sub-topic is requested
        if '.' in topic:
            parenttopic, subtopic = topic.rsplit('.', 1)
            parent = cls.__signals__.get(parenttopic) or \
                cls.__create_sig__(parenttopic, *args, **kwargs)

            sig = parent.subtopics.get(subtopic)
            if not sig:
                sig = super().__call__(topic)
                parent.subtopics[subtopic] = sig
                cls.__signals__[topic] = sig
        else:
            sig = super().__call__(topic, *args, **kwargs)
            cls.__signals__[topic] = sig

        return sig


class Signal(metaclass=SignalFactory):
    """ A signal can trigger one or more functions when it is emitted. """
    def __init__(self, topic=''):
        self.topic = topic
        self.subscribers = list()
        self.subtopics = dict()

    def __getitem__(self, subtopic):
        """ Return signal for specified sub-topic. """
        return self.subtopics.get(subtopic) or Signal(f'{self.topic}.{subtopic}')

    def emit(self, *args, **kwargs):
        """ Trigger the registered functions with passed arguments. """
        for sub in self.subscribers:
            sub(*args, **kwargs)

        for sub in self.subtopics.values():
            sub.emit(*args, **kwargs)

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
