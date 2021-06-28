""" BlueSky implementation of signals that can trigger one or more functions
    when a signal is emitted. """


class SignalFactory(type):
    ''' Factory meta-class for Signal objects in BlueSky. '''
    # Store all signal objects in process
    __signals = dict()

    def __call__(cls, name=''):
        ''' Factory function for Signal construction. '''
        # if no name is passed, return an anonymous Signal
        if not name:
            return super().__call__('anonymous')
        # Check if this Signal already exists
        sig = cls.__signals.get(name)
        if sig is None:
            # If it doesn't, create it
            sig = super().__call__(name)
            cls.__signals[name] = sig
        return sig

class Signal(metaclass=SignalFactory):
    """ A signal can trigger one or more functions when it is emitted. """
    def __init__(self, name=''):
        self.name = name
        self.__subscribers = []

    def get_subs(self):
        """ Return the list of subscribers to this signal. """
        return self.__subscribers

    def emit(self, *args, **kwargs):
        """ Trigger the registered functions with passed arguments. """
        for subs in self.__subscribers:
            subs(*args, **kwargs)

    def connect(self, func):
        """ Connect a new function to this signal. """
        self.__subscribers.append(func)

    def disconnect(self, func):
        """ Disconnect a function from this signal. """
        try:
            self.__subscribers.remove(func)
        except ValueError:
            print('Warning: function %s not removed '
                  'from signal %s'%(func,self))
