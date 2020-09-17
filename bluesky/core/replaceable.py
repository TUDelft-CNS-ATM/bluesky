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


class ReplaceableMeta(type):
    ''' Meta class to equip replaceable classes with a generator object. '''
    def __init__(cls, clsname, bases, attrs):
        super().__init__(clsname, bases, attrs)
        if clsname not in ['Replaceable', 'Entity']:
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
