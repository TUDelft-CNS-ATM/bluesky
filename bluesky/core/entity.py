''' Entity is a base class for all BlueSky singleton entities. '''
import inspect
from bluesky.core.replaceable import Replaceable, ReplaceableMeta


class Proxy:
    ''' Proxy class for BlueSky replaceable singleton entities. '''

    def __init__(self):
        # The reference object
        self.__dict__['_refobj'] = None
        # List of all proxied functions/methods
        self.__dict__['_proxied'] = list()

    def _selected(self):
        return self._refobj.__class__

    def _replace(self, refobj):
        # Replace our reference object
        self.__dict__['_refobj'] = refobj
        # First clear all proxied functions/methods
        for name in self._proxied:
            delattr(self, name)
        self._proxied.clear()
        # Copy all public functions/methods of reference object
        for name, value in inspect.getmembers(refobj, callable):
            if name[0] != '_':
                self.__dict__[name] = value
                self._proxied.append(name)

    def __getattr__(self, attr):
        return getattr(self._refobj, attr)

    def __setattr__(self, name, value):
        return setattr(self._refobj, name, value)


class EntityMeta(ReplaceableMeta):
    ''' Meta class to make replaceable classes singletons. '''
    def __init__(cls, clsname, bases, attrs):
        super().__init__(clsname, bases, attrs)
        cls._instance = None
        if clsname != 'Entity':
            # Initialise proxy object and stack command list for the first
            # descendant of Entity
            if cls._proxy is None:
                cls._proxy = Proxy()
            if cls._stackcmds is None:
                cls._stackcmds = dict()
            # Always update the stack command list by iterating over all stack commands
            for name, obj in inspect.getmembers(cls, lambda o: hasattr(o, '__stack_cmd__')):
                if name in cls._stackcmds:
                    # for subclasses reimplementing stack functions we keep only one
                    # Command object
                    if type(obj.__stack_cmd__) is not type(cls._stackcmds[name]):
                        raise TypeError(f'Error reimplementing {name}: '
                            f'A {type(cls._stackcmds[name]).__name__} cannot be '
                            f'reimplemented as a {type(obj.__stack_cmd__).__name__}')
                    obj.__stack_cmd__ = cls._stackcmds[name]
                else:
                    cls._stackcmds[name] = obj.__stack_cmd__
                cmd = obj.__stack_cmd__
                if not inspect.ismethod(cmd.callback) and inspect.ismethod(obj):
                    # Update callback of stack command with first bound
                    # method we encounter
                    cmd.callback = obj

    def __call__(cls, *args, **kwargs):
        ''' Object creation with proxy wrapping and ensurance of singleton
            behaviour. '''
        # check if the current instance is the same as the selected class
        if cls._proxy._selected() is not cls.selected():
            # If not, reset the proxy object to the selected implementation,
            # and create the instance if it hasn't been created yet
            if cls.selected()._instance is None:
                cls.selected()._instance = super().__call__(*args, **kwargs)
            # Update the object the proxy is referring to
            refobj = cls.selected()._instance
            cls._proxy._replace(refobj)
            # Update the stack commands of this class
            for name, cmd in cls._stackcmds.items():
                cmd.callback = getattr(refobj, name, cmd.notimplemented)
        return cls._proxy


class Entity(Replaceable, metaclass=EntityMeta):
    '''
        Super class for all BlueSky singleton entities.
    '''
    # Each Entity subclass keeps its own (single) instance
    _instance = None
    # Each first descendant of Entity has a proxy object that wraps the
    # currently selected instance
    _proxy = None
    # Each first descendant of Entity keeps a dict of all stack commands
    _stackcmds = None

    @classmethod
    def select(cls):
        ''' Select this class as generator. '''
        cls._replaceable._generator = cls
        _ = cls()

    @classmethod
    def instance(cls):
        ''' Return the current instance of this entity. '''
        return cls._proxy
