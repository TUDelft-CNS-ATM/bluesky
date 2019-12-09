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


class Proxy:
    ''' Proxy class for BlueSky replaceable singletons. '''
    def __init__(self):
        self._refobj = None
        self._proxied = list()

    def _replace(self, refobj):
        self._refobj = refobj
        for name in self._proxied:
            delattr(self, name)
        self._proxied.clear()
        # Copy all public methods of reference object
        for name, value in inspect.getmembers(refobj, callable):
            if name[0] != '_':
                setattr(self, name, value)
                self._proxied.append(name)

    def __getattr__(self, attr):
        return getattr(self._refobj, attr)


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
        if clsname != 'ReplaceableSingleton' and cls._proxy is None:
            cls._proxy = Proxy()

    def __call__(cls, *args, **kwargs):
        if type(cls._replaceable._instance) is not cls._replaceable._generator:
            cls._replaceable._instance = super().__call__(*args, **kwargs)
            cls._proxy._replace(cls._instance)
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
