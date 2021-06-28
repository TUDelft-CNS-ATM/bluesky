'''
    Provides Replaceable base class for classes in BlueSky that should allow
    replaceable implementations (in plugins) that can be selected at runtime.
'''

# Global dictionary of replaceable BlueSky classes
replaceables = dict()


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


class Replaceable:
    ''' Super class for BlueSky classes with replaceable implementations. '''
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
    def select(cls):
        ''' Select this class as generator. '''
        cls._baseimpl._generator = cls

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

    def __init_subclass__(cls, replaceable=True):
        ''' Register replaceable class bases. '''
        cls._generator = cls
        if replaceable and not hasattr(cls, '_baseimpl'):
            # Singleton Entitie derive from Replaceable, but have
            # the option to avoid being replaceable.
            cls._baseimpl = cls
            cls._default = ''
            replaceables[cls.__name__.upper()] = cls

    def __new__(cls, *args, **kwargs):
        ''' Replaced new to allow base class to construct selected derived instances. '''
        return object.__new__(cls._generator)
