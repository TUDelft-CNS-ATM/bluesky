''' Entity is a base class for all BlueSky singleton entities. '''
import inspect
from typing import Optional
from bluesky.core.base import Base
from bluesky.core.trafficarrays import TrafficArrays


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


def isproxied(obj):
    ''' Returns true if 'obj' is a proxied object. '''
    return isinstance(obj, Proxy)


def getproxied(obj):
    ''' Return wrapped proxy object if proxied, otherwise the original object. '''
    return obj.__dict__['_refobj'] if isinstance(obj, Proxy) else obj


class EntityMeta(type):
    ''' Meta class to make replaceable classes singletons. '''
    def __call__(cls, *args, **kwargs):
        ''' Object creation with proxy wrapping and ensurance of singleton
            behaviour. '''
        # Create singleton instance of this class if it doesn't exist yet
        if not cls.is_instantiated():
            super().__call__(*args, **kwargs)

        # When proxied, calling the base constructor returns the proxy object.
        # All derived constructors, and all non-proxied classes return the
        # actual instance.
        if cls._proxy and cls is cls.getbase():
            if getproxied(cls._proxy) is None:
                cls.select(cls._instance)
            return cls._proxy

        return cls._instance


class Entity(Base, TrafficArrays, metaclass=EntityMeta, skipbase=True):
    ''' Super class for BlueSky singleton entities (such as Traffic, Autopilot, ...). '''
    # Class variables that will be set for subclasses
    _proxy: Optional[Proxy]
    _instance: Optional['Entity']

    @classmethod
    def select(cls, instance=None):
        ''' Select this class as generator, and update
            function object references to those of the selected instance. '''
        instance = instance or cls._instance or cls()
        super().select(instance)
        if cls._proxy and cls._proxy._selected() is not cls:
            cls._proxy._replace(instance)


    @classmethod
    def is_instantiated(cls):
        ''' Returns true if the singleton of this class has already been instantiated. '''
        return cls._instance is not None

    @classmethod
    def instance(cls):
        ''' Return the current instance of this entity. '''
        return cls._proxy or cls._instance

    @classmethod
    def implinstance(cls):
        ''' Return the instance of this specific implementation. '''
        return cls._instance

    def __init__(self) -> None:
        super().__init__()
        cls = type(self)
        if cls._instance is None:
            cls._instance = self

    def __init_subclass__(cls, replaceable=False, skipbase=False):
        super().__init_subclass__(replaceable, skipbase)
        # When skipbase is True, an intermediate base class is currently defined,
        # and instance management shoud skip one step in the class tree.
        if skipbase:
            return

        # Each Entity subclass keeps its own (single) instance.
        cls._instance = None
        if not hasattr(cls, '_proxy'):
            # Each first descendant of replaceable Entities has a proxy object
            # that wraps the currently selected instance
            cls._proxy = Proxy() if replaceable else None
