''' Entity is a base class for all BlueSky singleton entities. '''
import inspect
from bluesky.core.replaceable import Replaceable
from bluesky.core.trafficarrays import TrafficArrays
from bluesky.core.simtime import timed_function


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
        # For non-replaceable Entities, _proxy is None,
        # just return the singleton instance
        if cls._proxy is None:
            if cls._instance is None:
                cls._instance = super().__call__(*args, **kwargs)
                # Update the stack commands of this class
                for name, cmd in cls._stackcmds.items():
                    cmd.callback = getattr(
                        cls._instance, name, cmd.notimplemented)
                # Update the timed functions of this class
                for name, timedfun in cls._timedfuns.items():
                    timedfun.callback = getattr(
                        cls._instance, name, timedfun.notimplemented)
            return cls._instance
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
            # Update the timed functions of this class
            for name, timedfun in cls._timedfuns.items():
                timedfun.callback = getattr(refobj, name, timedfun.notimplemented)
        return cls._proxy


class Entity(Replaceable, TrafficArrays, metaclass=EntityMeta, replaceable=False):
    ''' Super class for BlueSky singleton entities (such as Traffic, Autopilot, ...). '''
    @classmethod
    def select(cls):
        ''' Select this class as generator. '''
        super().select()
        _ = cls()

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

    def __init_subclass__(cls, replaceable=False, skipbase=False):
        super().__init_subclass__(replaceable)
        # When skipbase is True, an intermediate base class is currently defined,
        # and instance management shoud skip one step in the class tree.
        if skipbase:
            return
        # Each Entity subclass keeps its own (single) instance.
        cls._instance = None
        if not hasattr(cls, '_stackcmds'):
            # Each first descendant of Entity keeps a dict of all stack commands
            cls._stackcmds = dict()
            # All automatically-triggered timed methods
            cls._timedfuns = dict()
            # And all manually-triggered timed methods
            cls._manualtimedfuns = dict()

            # Each first descendant of replaceable Entities has a proxy object
            # that wraps the currently selected instance
            cls._proxy = Proxy() if replaceable else None

        for name, obj in inspect.getmembers(cls):
            # Always update the stack command list by iterating over all stack commands
            if hasattr(obj, '__stack_cmd__'):
                if name not in cls._stackcmds:
                    cls._stackcmds[name] = obj.__stack_cmd__
                cmd = obj.__stack_cmd__
                if not inspect.ismethod(cmd.callback) and inspect.ismethod(obj):
                    # Update callback of stack command with first bound method we encounter
                    cmd.callback = obj
            # Similarly also always update the timed function lists
            timer = cls._manualtimedfuns.get(name)
            if timer and not hasattr(obj, '__manualtimer__'):
                # Reimplemented timed functions in derived classes should share
                # timers with their originals
                setattr(cls, name, timed_function(obj, manual=True, timer=timer))
            elif hasattr(obj, '__manualtimer__'):
                cls._manualtimedfuns[name] = obj.__manualtimer__
            elif hasattr(obj, '__timedfun__'):
                if name not in cls._timedfuns:
                    cls._timedfuns[name] = obj.__timedfun__
                timedfun = obj.__timedfun__
                if not inspect.ismethod(timedfun.callback) and inspect.ismethod(obj):
                    # Update callback of the timed function with first bound method we encounter
                    timedfun.callback = obj
