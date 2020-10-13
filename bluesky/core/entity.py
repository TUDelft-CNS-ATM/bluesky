''' Entity is a base class for all BlueSky singleton entities. '''
import inspect
from bluesky.core.replaceable import Replaceable
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
    def instance(cls):
        ''' Return the current instance of this entity. '''
        return cls._proxy

    def __init_subclass__(cls, replaceable=False):
        super().__init_subclass__(replaceable)
        # Each Entity subclass keeps its own (single) instance
        cls._instance = None
        if not hasattr(cls, '_stackcmds'):
            # Each first descendant of Entity keeps a dict of all stack commands
            cls._stackcmds = dict()
            # And all timed methods
            cls._timedfuns = dict()

            # Each first descendant of replaceable Entities has a proxy object
            # that wraps the currently selected instance
            cls._proxy = Proxy() if replaceable else None

        # Always update the stack command list by iterating over all stack commands
        for name, obj in inspect.getmembers(cls, lambda o: hasattr(o, '__stack_cmd__')):
            if name not in cls._stackcmds:
                cls._stackcmds[name] = obj.__stack_cmd__
            cmd = obj.__stack_cmd__
            if not inspect.ismethod(cmd.callback) and inspect.ismethod(obj):
                # Update callback of stack command with first bound method we encounter
                cmd.callback = obj
        # Similarly also always update the timed function list
        for name, obj in inspect.getmembers(cls, lambda o: hasattr(o, '__timedfun__')):
            if name not in cls._timedfuns:
                cls._timedfuns[name] = obj.__timedfun__
            timedfun = obj.__timedfun__
            if not inspect.ismethod(timedfun.callback) and inspect.ismethod(obj):
                # Update callback of the timed function with first bound method we encounter
                timedfun.callback = obj
