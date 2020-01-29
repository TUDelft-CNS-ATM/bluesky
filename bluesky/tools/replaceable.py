'''
    Provides Replaceable base class for classes in BlueSky that should allow
    replaceable implementations (in plugins) that can be selected at runtime.
'''
import inspect


def reset():
    ''' Reset all replaceables to their default implementation. '''
    for base in Replaceable._replaceables.values():
        base.select()


def select_implementation(basename='', implname=''):
    ''' Stack function to select an implementation for the construction of
        objects of the class corresponding to basename. '''
    if not basename:
        return True, 'Replaceable classes in Bluesky:\n' + \
            ', '.join(Replaceable._replaceables)
    base = Replaceable._replaceables.get(basename.upper(), None)
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


def check_method(fun):
    ''' Check if passed function is a method of a ReplaceableSingleton. '''
    if inspect.ismethod(fun) and isinstance(fun.__self__, ReplaceableSingleton):
        return fun.__self__._proxy._methodproxy(fun)
    return fun


class Methodproxy:
    ''' Proxy class for methods of replaceable singletons. '''

    def _notimplemented(self, *args, **kwargs):
        return False, f'The current {self._basename} implementation doesn\'t' +\
            f'provide this function (function was originally declared in {self._origimpl})'

    def __init__(self, fun):
        self._fun = fun
        self._origimpl = fun.__self__.__class__.__name__
        self._basename = fun.__self__.__class__._replaceable.__name__
        self.__defaults__ = fun.__defaults__

    def __call__(self, *args, **kwargs):
        return self._fun(*args, **kwargs)

    def _update(self, fun):
        self._fun = fun
        self.__defaults__ = fun.__defaults__

    def _reset(self):
        self._fun = self._notimplemented


class Proxy:
    ''' Proxy class for BlueSky replaceable singletons. '''
    def __init__(self):
        self.__dict__['_refobj'] = None
        self.__dict__['_proxied'] = list()
        self.__dict__['_wrappedmethods'] = dict()

    def _selected(self):
        return self._refobj.__class__

    def _methodproxy(self, fun):
        ret = Methodproxy(fun)
        if fun.__name__ in self._proxied:
            delattr(self, fun.__name__)
            self._proxied.remove(fun.__name__)
        self._wrappedmethods[fun.__name__] = ret
        self.__dict__[fun.__name__] = ret
        return ret

    def _replace(self, refobj):
        # Replace our reference object
        self.__dict__['_refobj'] = refobj
        for name in self._proxied:
            delattr(self, name)
        self._proxied.clear()
        wrappedmethods = dict(self._wrappedmethods)
        # Copy all public methods of reference object
        for name, value in inspect.getmembers(refobj, callable):
            if name[0] != '_':
                wrapped = wrappedmethods.pop(name, None)
                if wrapped is None:
                    self.__dict__[name] = value
                    self._proxied.append(name)
                else:
                    wrapped._update(value)
        # Clear any remaining method wrappers
        for wrapped in wrappedmethods.values():
            wrapped._reset()

    def __getattr__(self, attr):
        return getattr(self._refobj, attr)

    def __setattr__(self, name, value):
        return setattr(self._refobj, name, value)


class ReplaceableMeta(type):
    ''' Meta class to equip replaceable classes with a generator object. '''
    def __init__(cls, clsname, bases, attrs):
        super().__init__(clsname, bases, attrs)
        if clsname not in ['Replaceable', 'ReplaceableSingleton']:
            cls._generator = cls
            # Keep track of the base replaceable class
            if cls._replaceable is None:
                # Register this class as a base replaceable class
                cls._replaceable = cls
                cls._replaceables[clsname.upper()] = cls

    def __call__(cls, *args, **kwargs):
        ret = object.__new__(cls._generator)
        ret.__init__(*args, **kwargs)
        return ret


class ReplaceableSingletonMeta(ReplaceableMeta):
    ''' Meta class to make replaceable classes singletons. '''
    def __init__(cls, clsname, bases, attrs):
        super().__init__(clsname, bases, attrs)
        cls._instance = None
        if clsname != 'ReplaceableSingleton' and cls._proxy is None:
            cls._proxy = Proxy()

    def __call__(cls, *args, **kwargs):
        # check if the current instance is the same as the selected class
        if cls._proxy._selected() is not cls.selected():
            # If not, reset the proxy object to the selected implementation,
            # and create the instance if it hasn't been created yet
            if cls.selected()._instance is None:
                cls.selected()._instance = super().__call__(*args, **kwargs)
            cls._proxy._replace(cls.selected()._instance)
        return cls._proxy


class Replaceable(metaclass=ReplaceableMeta):
    ''' Super class for BlueSky classes with replaceable implementations. '''
    _replaceables = dict()
    _replaceable = None
    _generator = None

    @classmethod
    def select(cls):
        ''' Select this class as generator. '''
        cls._replaceable._generator = cls

    @classmethod
    def selected(cls):
        ''' Return the selected implementation. '''
        return cls._replaceable._generator

    @classmethod
    def derived(cls):
        ''' Recursively find all derived classes of cls. '''
        ret = {cls.__name__.upper(): cls}
        for sub in cls.__subclasses__():
            ret.update(sub.derived())
        return ret


class ReplaceableSingleton(Replaceable, metaclass=ReplaceableSingletonMeta):
    '''
        Super class for BlueSky singleton classes with replaceable
        implementations.
    '''
    _instance = None
    _proxy = None

    @classmethod
    def select(cls):
        ''' Select this class as generator. '''
        cls._replaceable._generator = cls
        _ = cls()

    @classmethod
    def instance(cls):
        return cls._proxy
