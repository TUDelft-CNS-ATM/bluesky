""" BlueSky implementation of signals that can trigger one or more functions
    when a signal is emitted. """

class Signal:
    """ A signal can trigger one or more functions when it is emitted. """
    def __init__(self):
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
