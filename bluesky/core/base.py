'''
    Provides BlueSky derived classes in BlueSky with the following functionality:
    - allow replaceable implementations (in plugins) that can be selected at runtime.
    - allow class and instance methods to be decorated as
      - stack functions
      - network subscribers
      - shared state subscribers
      - timed functions
      - signal subscribers
'''
import inspect
from typing import Dict, Optional, Type, ClassVar

from bluesky.core.funcobject import FuncObject

# Global dictionary of replaceable BlueSky classes
replaceables: Dict[str, Type['Base']] = dict()


def reset():
    ''' Reset all replaceables to their default implementation. '''
    for base in replaceables.values():
        base.selectdefault()


def select_implementation(basename='', implname=''):
    ''' Stack function to select an implementation for the construction of
        objects of the class corresponding to basename. '''
    if not basename:
        return True, 'Replaceable classes in Bluesky:\n' + \
            ', '.join(replaceables)
    base = replaceables.get(basename.upper(), None)
    if not base:
        return False, f'Replaceable {basename} not found.'
    impls = base.derived()
    if not implname:
        return True, f'Current implementation for {basename}: {base._generator.__name__}\n' + \
            f'Available implementations for {basename}:\n' + \
            ', '.join(impls)

    impl = impls.get(base.__name__ if implname == 'BASE' else implname)
    if not impl:
        return False, f'Implementation {implname} not found for replaceable {basename}.'
    impl.select()
    return True, f'Selected implementation {implname} for replaceable {basename}'


class Base:
    '''
        Base: Provides BlueSky derived classes in BlueSky with the following functionality:
        - allow replaceable implementations (in plugins) that can be selected at runtime.
        - allow class and instance methods to be decorated as
          - stack functions
          - network subscribers
          - shared state subscribers
          - timed functions
          - signal subscribers
    '''
    # Class variables that will be set for subclasses
    _baseimpl: ClassVar[Type['Base']]
    _default: ClassVar[str]
    _generator: ClassVar[Type['Base']]
    _selinstance: ClassVar[Optional['Base']]
    __func_objects__: ClassVar[Dict[str, FuncObject]]

    @classmethod
    def setdefault(cls, name):
        ''' Set a default implementation. '''
        impl = cls._baseimpl.derived().get(name.upper())
        if impl:
            cls._baseimpl._default = name.upper()
            cls._baseimpl._generator = impl

    @classmethod
    def getdefault(cls):
        ''' Get the default implementation. '''
        default = cls._baseimpl._default
        return cls._baseimpl.derived().get(default) if default else cls._baseimpl

    @classmethod
    def getbase(cls):
        ''' Get the base implementation. '''
        return cls._baseimpl

    @classmethod
    def name(cls):
        ''' Return the name of this implementation. '''
        return cls.__name__.upper()

    @classmethod
    def selectdefault(cls):
        ''' Select the default implementation. '''
        base = cls._baseimpl
        base.derived().get(base._default, base).select()

    @classmethod
    def select(cls, instance=None):
        ''' Select instance class or caller class as generator,
            and update known function object references. '''
        if instance is not None:
            cls._baseimpl._generator = type(instance)
            cls._baseimpl._selinstance = instance
            target = instance
        else:
            cls._baseimpl._generator = cls
            target = cls

        for name, fobj in target.__func_objects__.items():
            if not inspect.ismethod(fobj.func): # TODO: doesn't this disallow e.g., selecting different stack function implementations?
                fobj.update(getattr(target, name, None))

    @classmethod
    def selected(cls):
        ''' Return the selected implementation. '''
        return cls._baseimpl._generator

    @classmethod
    def derived(cls):
        ''' Recursively find all derived classes of cls. '''
        ret = {cls.__name__.upper(): cls}
        for sub in cls.__subclasses__():
            ret.update(sub.derived())
        return ret

    def __init_subclass__(cls, replaceable=True, skipbase=False):
        ''' Register replaceable class bases. '''
        if not skipbase:
            cls._generator = cls
            # Add base functionality
            # Each first implementation descendant of Base
            # is referenced, and keeps a dict of all
            # function-wrapping objects
            if not hasattr(cls, '_baseimpl'):
                cls._baseimpl = cls
                cls._selinstance = None
                cls.__func_objects__ = dict()
    
                # Singleton Entities derive from Replaceable, but have
                # the option to avoid being replaceable.
                if replaceable:
                    cls._default = ''
                    replaceables[cls.__name__.upper()] = cls

            # Always update the function object list by iterating over all items
            for name, func in inspect.getmembers(cls):
                ufunc = inspect.unwrap(func)
                fobj = cls.__func_objects__.get(name)
                if fobj or hasattr(ufunc, '__func_object__'):
                    if fobj is None:
                        fobj = ufunc.__func_object__
                        cls.__func_objects__[name] = fobj
                    else:
                        setattr(ufunc, '__func_object__', fobj)
                    if not inspect.ismethod(fobj.func) and inspect.ismethod(func):
                        # Update callback of function object with first bound method we encounter
                        fobj.update(func)
        return super().__init_subclass__()

    def __init__(self) -> None:
        super().__init__()
        # Make sure that the first constructed instance of the selected class is also selected, 
        # even if select() isn't called explicitly
        cls = type(self)
        if cls._selinstance is None and cls is cls.selected():
            cls.select(self)

    def __new__(cls, *args, **kwargs):
        ''' Replaced new to allow base class to construct selected derived instances. '''
        # Calling the base constructor should return an instance of the
        # selected class. Explicitly calling a subclass constructor should
        # return an instance of that class
        generator = cls._generator if cls is cls.getbase() else cls

        return super().__new__(generator)
